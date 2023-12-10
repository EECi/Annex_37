"""Train DMS models required to run experiments."""

import os
import sys
import csv
import time
import glob
from models.dms.predictor import Predictor
import torch
import numpy as np
import random


def get_mparams(model_type, L ,T):

    if model_type == 'linear':
        mparam_dict = {'all': {'model_name': 'vanilla',
                            'mparam': {'L': L,
                                        'T': T,
                                        'layers': (),
                                        }
                            }
                    }

    elif model_type == 'resmlp':
        mparam_dict = {'all': {'model_name': 'resmlp',
                            'mparam': {'L': L,
                                        'T': T,
                                        'feature_sizes': (168,),
                                        }
                            }
                    }

    elif model_type == 'conv':
        mparam_dict = {'all': {'model_name': 'conv',
                            'mparam': {'L': L,
                                        'T': T,
                                        'channels': (5,),
                                        'kernel_sizes': (6,),
                                        'output_kernel_size': 12
                                        }
                            }
                    }
    
    return mparam_dict


if __name__ == '__main__':

    # Run using
    # for ($m = 0; $m -le 2; $m++) {for ($rd = 0; $rd -le 5; $rd++) {python -m experiments.train_DMS_models $m $rd}}
    # ==================================================================================================

    m = int(sys.argv[1])
    rd = int(sys.argv[2])

    model_types = ['linear','resmlp','conv']
    data_lengths = ['baseline','rd4y','rd2y','rd1y','rd6m','rd3m']

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    L = 168
    T = 48

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]

    mtype = model_types[m]
    dl = data_lengths[rd]

    if dl == 'baseline':
        expt_name =  os.path.join('analysis',mtype)
        dataset_dir = os.path.join('data','analysis')
    else:
        expt_name =  os.path.join('analysis',mtype+'-%s'%dl)
        dataset_dir = os.path.join('data','analysis','reduced',dl)

    start = time.time()
    predictor = Predictor(get_mparams(mtype,L,T), UCam_ids, L, T, expt_name, load=False)
    predictor.train(patience=100, max_epoch=500, dataset_dir=dataset_dir)
    end = time.time()

    print("Train time:", end-start)

    with open(os.path.join('models','dms','resources',expt_name,'training_time.csv'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Create & train time (s)', end-start])

    # clear learning rate checkpoints from current directory
    lr_checkpoint_list = glob.glob('.lr_find*')
    for lr_checkpoint in lr_checkpoint_list:
        os.remove(lr_checkpoint)
