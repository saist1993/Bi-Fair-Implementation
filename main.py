# Nothing fancy. Just a skeleton implementation of bi-fair. Then it can be optimized for more complex task.
import torch
import random
import higher # don't exactly know why is this needed. But most likley it is easy to copy model params.
import numpy as np
from tqdm.auto import tqdm
import sklearn.metrics as sk
import torch.nn.functional as F
from bifair_utils import *
from utils.fairness_functions import *
from models import SimpleNonLinear, SimpleLinear
from create_data import SimpleAdvDatasetReader, EncodedEmoji




def run(inner_wd, outer_wd, es_tol, weight_len, fair_lambda, f_name, dataset_name, batch_size, model_type, device, T_outloop, T_in, chkpt, fairness_function, seed):
    print(f"seed is {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # artifcats for creating training data.
    params = {
        'batch_size': batch_size,
        'fairness_iterator': 'custom_3', # dummy not needed
        'fair_grad_newer': False,
        'fairness_iterator': 'some dummy sruff'
    }

    if dataset_name == 'adult':
        dataset_object = SimpleAdvDatasetReader(dataset_name, **params)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_data_metadata = dataset_object.run()
    elif dataset_name == 'encoded_emoji':
        dataset_object = EncodedEmoji(dataset_name, **params)
        vocab, number_of_labels, number_of_aux_labels, iterators, other_data_metadata = dataset_object.run()
    else:
        raise NotImplementedError


    # setting up input and output dim for model which varies based on the dataset.
    output_dim = 1 # output dim for the model. In our model it is 2, but here they use 1 and then a different kind of loss function.
    for items in iterators[0]['train_iterator']: # input dim calculated using dataset iterator info.
        item = items
        break
    input_dim = item['input'].shape[1]

    # setting up model params. Mostly old code, thus this complex setup.
    model_arch = {
        'encoder': {
            'input_dim': input_dim,
            'output_dim': 1
        }
    }
    model_params = {
        'model_arch': model_arch,
        'device': device
    }

    # finally between simple_non_linear and siple_linear.
    if model_type == 'simple_non_linear':
        model = SimpleNonLinear(model_params)
    elif model_type == 'simple_linear':
        model = SimpleLinear(model_params)
    else:
        raise NotImplementedError




    opt = torch.optim.Adam(model.parameters(), weight_decay=inner_wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=(es_tol//2 - 1), verbose=True)
    es = EarlyStopping(patience=es_tol, verbose=True, path=f'{f_name}.pt')

    # setting up the weight params equation.
    # Ideally, this should be of the length of training data. However, in appendix and in their code base they show methods
    # where you can use 4, or 8, or somethign else as weights.
    # They have a good explaination of how  weight of len 8 can work. Check their appendix section.
    w = torch.rand( (weight_len, 1), requires_grad=False, device=device)
    w /= torch.norm(w, p=1)
    w.requires_grad=True
    opt_weights = torch.optim.Adam([w], weight_decay=outer_wd)






    """
    The training loop looks like the following 
    
    for iterations in total_outerloop_iterations:
        for train_batch_size in range(total_number_of_inner_loop):
            > do stuff
        > over alll validation set
        > do stuff
    
    """
    # setting up dataloaders
    v_generator = inf_dataloader(iterators[0]['valid_iterator'], device)
    tr_generator = inf_dataloader(iterators[0]['train_iterator'], device)

    tr_loss_sum = 0.0
    # now the main training loop


    for t_out in tqdm(range(T_outloop)):
        with higher.innerloop_ctx(model, opt) as (fmodel, diffopt): # don't know why is this needed.
            for _ in range(T_in):
                idx, items = next(tr_generator) # to check what items contain -> see create data SimpleAdvDatasetReader. collate function.
                # diffopt.zero_grad() # this is not in the original codebase. According to documentation this is not needed.
                t_output = fmodel(items) # forward pass from the model
                tr_loss = F.binary_cross_entropy_with_logits(t_output['prediction'].squeeze(), items['labels'].squeeze().float(), reduction='none')
                tr_loss_sum += tr_loss.mean().item()
                tr_loss = get_weighted_loss(logits=t_output['prediction'].squeeze(), weights=w, items=items, loss=tr_loss)
                diffopt.step(tr_loss.mean())

            idx, v_items = next(v_generator)
            v_output = fmodel(v_items)
            v_util_loss = F.binary_cross_entropy_with_logits(v_output['prediction'].squeeze(), v_items['labels'].squeeze().float(), reduction='none')
            v_fair_loss = get_fairness_loss(logits=v_output['prediction'].squeeze(), items=v_items, loss=v_util_loss)
            v_total_loss = v_util_loss.mean() + fair_lambda * v_fair_loss
            opt_weights.zero_grad()
            w.grad = torch.autograd.grad(v_total_loss, w)[0] # looks like we are diff wrt model params before the trian epochs
            opt_weights.step()
            with torch.no_grad():
                w /= torch.norm(w, p=1)
                # w.clip_(min=0)
                model.load_state_dict(fmodel.state_dict())
        if (t_out + 1) %  chkpt == 0: #
            with torch.no_grad():
                model.eval()
                ep = t_out // len(iterators[0]['train_iterator'])
                tr_loss_sum /= (len(iterators[0]['train_iterator']) * chkpt * T_in)
                print(f'Loss/Train/Loss',  tr_loss_sum, ep)
                t, v_metrics = infer(model, iterators[0]['valid_iterator'], device, fairness_function, threshold=None)
                print(calculate_final_fairness(v_metrics['interm_group_fairness']))
                print(v_metrics)
                stop_loss = v_metrics['loss'] + fair_lambda * v_metrics['fairness_loss']
                es(stop_loss, model)
                if es.early_stop:
                    print("Early stopping")
                    break
                scheduler.step(stop_loss)
                model.train()
                tr_loss_sum = 0


    with torch.no_grad():
        model.eval()
        v_threshold, v_metrics = infer(model, iterators[0]['valid_iterator'], device, fairness_function, threshold=None)
        _, t_metrics = infer(model, iterators[0]['test_iterator'], device, fairness_function, threshold=v_threshold)
        print(f"validation threshold:{v_threshold}")
        print(f"validation metrics")
        print(v_metrics)
        print(f"test metrics")
        print(t_metrics)
        print(calculate_final_fairness(t_metrics['interm_group_fairness']))

        def transform_ouputs(input_dict):
            new_dict = {}
            for key, value in input_dict.items():
                try:
                    new_dict[key] = value.tolist()
                except:
                    new_dict[key] = value
            return new_dict


        final_eval = {
            'validation_metrics': transform_ouputs(v_metrics),
            'test_metrics': transform_ouputs(t_metrics)
        }

        return final_eval


if __name__ == '__main__':
    dataset_name = 'adult'
    batch_size = 64
    model_type = 'simple_non_linear'
    device = 'cpu'
    T_outloop = 1000
    T_in = 5
    chkpt = 10  # test every 10 iterations. T_in*chk_pt data points.
    fairness_function = accuracy_parity

    # setting up the hyper param for BiFair approach
    inner_wd = 0.001  # still need to set this --> for  wd in 0 0.0001 0.001
    outer_wd = 0  # No idea what to set here
    es_tol = 1  # still need to set this
    weight_len = 8  # still need to set this
    fair_lambda = 0.5  # 0.5, 1, 2, 4
    es_tol = 10
    f_name = 'bifair'

    final_eval = run(inner_wd, outer_wd, es_tol, weight_len, fair_lambda, f_name, dataset_name, batch_size, model_type, device,
        T_outloop, T_in, chkpt, fairness_function, 256)

    print(final_eval)