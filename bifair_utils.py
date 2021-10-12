import torch
import numpy as np
from tqdm.auto import tqdm
import sklearn.metrics as sk
import torch.nn.functional as F

def inf_dataloader(dataloader, device):
    '''
        item = {
            'labels': labels,
            'input': input,
            'lengths': lengths,
            'aux': aux
        }
    '''

    while True:
        for idx, items in enumerate(dataloader): #
            for key in items.keys():
                items[key] = items[key].to(device)
            yield idx, items


def get_weighted_loss(logits, weights, items, loss):
    """

    All in all idea is to multiple weights of each example with corresponding loss function.
    fav, unfav - > s=1, s=0
    p, up -> y=1, y=0

    4 groups
    {s=0, y=0}, {s=1, y=0}, {s=0, y=1}, {s=1, y=1}
    :return:

    y = items['labels']
    s = items['aux']
    """
    probs = logits.sigmoid()
    thresholds = torch.linspace(0, 1, (len(weights) // 4) + 1)  # even spaced out
    p_fav_idx = torch.logical_and(items['labels']==1, items['aux']==1)
    p_unfav_idx = torch.logical_and(items['labels'] == 1, items['aux'] == 0)
    up_fav_idx = torch.logical_and(items['labels']==0, items['aux']==1)
    up_unfav_idx = torch.logical_and(items['labels'] == 0, items['aux'] == 0)

    # for i in items:
    #     y = i['labels']
    #     s = i['aux']
    #
    #     if y == 1 and s == 1:
    #         p_fav_idx.append(1)
    #         p_unfav_idx.append(0)
    #         up_fav_idx.append(0)
    #         up_unfav_idx.append(0)
    #     if y == 1 and s == 0:
    #         p_fav_idx.append(0)
    #         p_unfav_idx.append(1)
    #         up_fav_idx.append(0)
    #         up_unfav_idx.append(0)
    #     if y == 0 and s == 1:
    #         p_fav_idx.append(0)
    #         p_unfav_idx.append(0)
    #         up_fav_idx.append(1)
    #         up_unfav_idx.append(0)
    #     if y == 0 and s == 0:
    #         p_fav_idx.append(0)
    #         p_unfav_idx.append(0)
    #         up_fav_idx.append(0)
    #         up_unfav_idx.append(1)
    #
    # p_fav_idx, p_unfav_idx, up_fav_idx, up_unfav_idx = \
    #     torch.tensor(p_fav_idx), torch.tensor(p_unfav_idx), torch.tensor(up_fav_idx), torch.tensor(up_unfav_idx)

    for i in range(len(thresholds)-1):
        p_fav_idx_th = torch.logical_and(p_fav_idx, (probs >= thresholds[i]) & (probs < thresholds[i + 1]))
        p_unfav_idx_th = torch.logical_and(p_unfav_idx, (probs >= thresholds[i]) & (probs < thresholds[i + 1]))
        up_fav_idx_th = torch.logical_and(up_fav_idx, (probs >= thresholds[i]) & (probs < thresholds[i + 1]))
        up_unfav_idx_th = torch.logical_and(up_unfav_idx, (probs >= thresholds[i]) & (probs < thresholds[i + 1]))

        loss[p_fav_idx_th] *= weights[i * 4]
        loss[p_unfav_idx_th] *= weights[i * 4 + 1]
        loss[up_fav_idx_th] *= weights[i * 4 + 2]
        loss[up_unfav_idx_th] *= weights[i * 4 + 3]

    return loss


def get_fairness_loss(logits, items, loss=None):
    p_fav_idx = torch.logical_and(items['labels'] == 1, items['aux'] == 1)
    p_unfav_idx = torch.logical_and(items['labels'] == 1, items['aux'] == 0)
    up_fav_idx = torch.logical_and(items['labels'] == 0, items['aux'] == 1)
    up_unfav_idx = torch.logical_and(items['labels'] == 0, items['aux'] == 0)

    # probs = logits.sigmoid()

    fav_diff = torch.abs(loss[up_fav_idx].mean() - loss[p_fav_idx].mean())
    return fav_diff


def generate_predictions(model, iterator, device):
    all_preds = []
    fairness_all_aux, all_y = [], []

    with torch.no_grad():
        for items in tqdm(iterator):

            # setting up the device
            for key in items.keys():
                items[key] = items[key].to(device)

            fairness_all_aux.append(items['aux'])  # aux label.
            all_y.append(items['labels'])  # main task label.

            items['gradient_reversal'] = False
            output = model(items)
            predictions = output['prediction']
            all_preds.append(predictions.squeeze())

    # flattening all_preds
    all_preds = torch.cat(all_preds, out=torch.Tensor(len(all_preds), all_preds[0].shape[0])).to(device)

    fairness_all_aux = torch.cat(fairness_all_aux, out=torch.Tensor(len(fairness_all_aux), fairness_all_aux[0].shape[0])).to(device)
    all_y = torch.cat(all_y, out=torch.Tensor(len(all_y), all_y[0].shape[0])).to(device)
    total_no_aux_classes, total_no_main_classes = len(torch.unique(fairness_all_aux)), len(torch.unique(all_y))


    return all_preds, fairness_all_aux, all_y, total_no_aux_classes, total_no_main_classes


def find_opt_threshold(logits, lbls):
    probs = logits.sigmoid()

    num_thresh = 200
    bacc_arr = torch.zeros(num_thresh)
    thresholds = torch.linspace(0.01, 0.99, num_thresh)
    for idx, thresh in enumerate(thresholds):
        preds = probs > thresh
        bacc_arr[idx] = sk.balanced_accuracy_score(lbls.cpu(), preds.cpu())

    best_ind = torch.argmax(bacc_arr).item()
    return thresholds[best_ind]


def get_metrics(logits, lbls, items, fairness_function, device, threshold=0.5, preds=None):
    loss = F.binary_cross_entropy_with_logits(logits, lbls, reduction='none')
    probs = logits.sigmoid()
    if preds is None:
        preds = probs > threshold

    avg_loss = loss.mean().item()
    acc = sk.accuracy_score(lbls.cpu(), preds.cpu())
    bacc = sk.balanced_accuracy_score(lbls.cpu(), preds.cpu())
    utility_metrics = {
        'loss': avg_loss,
        'acc': acc,
        'bacc': bacc,
    }
    # fairness_loss = get_fairness_loss(probs, lbls, metas, loss=loss)


    fairness_loss = get_fairness_loss(logits, items, loss=loss) # needs to be generalized
    fairness_metrics = {}
    # fairness_metrics = get_fairness_metrics(preds, lbls, metas, loss)
    fairness_metrics['fairness_loss'] = fairness_loss.item()
    interm_group_fairness, interm_fairness_lookup, left_hand_matrix, sub_group_acc_matrix = fairness_function(preds, items['labels'], items['aux'], device, items['total_no_main_classes'], items['total_no_aux_classes'], epsilon=0.0)
    fairness_metrics['interm_group_fairness'] = interm_group_fairness
    # fairness_metrics['interm_fairness_lookup'] = interm_fairness_lookup
    fairness_metrics['left_hand_matrix'] = left_hand_matrix
    fairness_metrics['sub_group_acc_matrix'] = sub_group_acc_matrix
    # fairness_metrics['generalized fnr'] = probs.mean().item()
    # fairness_metrics['generalized fpr'] = (1 - probs.mean()).item()
    utility_metrics['total_loss'] = avg_loss + fairness_loss.item()

    return {**utility_metrics, **fairness_metrics}


def infer(model, iterator, device, fairness_function, threshold=0.5):
    logits, aux, lbls, total_no_aux_classes, total_no_main_classes = generate_predictions(model, iterator, device)
    if threshold is None:
        threshold = find_opt_threshold(logits, lbls)
    items = {
        'labels':lbls,
        'aux': aux,
        'total_no_main_classes': total_no_main_classes,
        'total_no_aux_classes': total_no_aux_classes
    }
    return threshold, get_metrics(logits, lbls, items, fairness_function, device, threshold, None)



def calculate_final_fairness(data_dict):
    data_matrix = []
    for key,value in data_dict.items():
        _temp = []
        for _key,_value in value.items():
            _temp.append(_value)
        data_matrix.append(_temp)

    all_vs = [i for d in data_matrix for i in d]
    mean_abs = np.mean([abs(i) for i in all_vs])
    maximum = max(all_vs)
    minimum = min(all_vs)
    temp = {
        'mean_abs': mean_abs,
        'maximum': maximum,
        'minimum': minimum
    }
    return temp

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """ Taken from https://github.com/Bjarten/early-stopping-pytorch """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

