import main
import torch
import argparse
from utils.fairness_functions import *


import uuid
import logging
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', '-name', help="log file name", type=str)
    parser.add_argument('--dataset_name', '-dataset', help="name of the dataset", type=str)
    parser.add_argument('--seed', '-seed', nargs="*", help="--seed 2 4 8 16 32 64 128 256 512 42", type=int)
    parser.add_argument('--model', '-model', help="end of adv scale 1.0", type=str)
    parser.add_argument('--fairness_function', '-fairness_function', help="accuracy_parity", type=str)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    create_dir = lambda dir_location: Path(dir_location).mkdir(parents=True, exist_ok=True)


    seeds = args.seed
    if args.fairness_function == 'accuracy_parity':
        fairness_function = accuracy_parity
    elif args.fairness_function == 'equal_opportunity':
        fairness_function = equal_opportunity
    elif args.fairness_function == 'equal_odds':
        fairness_function = equal_odds
    else:
        raise NotImplementedError

    chkpt = 10
    device = 'cpu'
    T_outloop = 1000
    dataset_name = args.dataset_name
    model_type = args.model

    logs_dir = Path('logs/bi_fair')
    create_dir(logs_dir)

    # create dataset dir in logs_dir
    dataset_dir = logs_dir / Path(dataset_name)
    create_dir(dataset_dir)
    log_file_name = str(dataset_dir / Path(args.log_name)) + '.log'

    # logger init
    logging.basicConfig(filename=log_file_name,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info(f"logging for {log_file_name}")
    logger = logging.getLogger('main')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


    def transform_ouputs(input_dict):
        new_dict = {}
        for key, value in input_dict.items():
            try:
                new_dict[key] = value.tolist()
            except:
                new_dict[key] = value
        return new_dict


    counter = 0
    for seed in seeds:
        for bs in [128,256,512]: # batch size
            for inner_wd in [0.0, 0.001]: # weight decay of adam
                for fair_lambda in [0.5, 1, 2, 4]: # fairness lambda
                    for weight_len in [8]:
                        for T_in in [5,25,50]: # Inner loop length
                            print(f"****************************{counter}*********************************")
                            counter = counter + 1
                            batch_size = bs
                            # setting up the hyper param for BiFair approach
                            outer_wd = inner_wd  # No idea what to set here
                            es_tol = 10  # still need to set this
                            es_tol = 10
                            f_name = 'bifair'
                            final_eval = main.run(inner_wd, outer_wd, es_tol, weight_len, fair_lambda, f_name, dataset_name, batch_size, model_type,
                                             device,
                                             T_outloop, T_in, chkpt, fairness_function,args.fairness_function, seed)
                            aux_data = {
                                'bs':bs,
                                'inner_wd': inner_wd,
                                'fair_lambda': fair_lambda,
                                'weight_len': weight_len,
                                'T_in': T_in,
                                'seed': seed,
                                'fairness_function': args.fairness_function,
                                'device': device,
                                'T_outloop': T_outloop,
                                'dataset_name': dataset_name,
                                'model_type': model_type,
                                'chkpt': chkpt,
                                'seed': seed
                            }
                            final_eval['aux_data'] = aux_data
                            print(final_eval)
                            logger.info(f"new run info: {transform_ouputs(final_eval)}".replace('\n', ''))