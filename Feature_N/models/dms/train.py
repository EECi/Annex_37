import os
import csv
import time
import glob
from models.dms.predictor import Predictor
import torch
import numpy as np
import random

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
L = 168
T = 48

for x in range(3):

    if x == 0:
        # linear experiments ---------------------------------------------------------------------------------------------------
        expt_name = f'linear_{seed}'
        mparam_dict = {'all': {'model_name': 'vanilla',
                            'mparam': {'L': L,
                                        'T': T,
                                        'layers': (),
                                        }
                            }
                    }
        # ----------------------------------------------------------------------------------------------------------------------

    if x == 1:
        # resmlp experiments ---------------------------------------------------------------------------------------------------
        expt_name = f'resmlp_{seed}'
        mparam_dict = {'all': {'model_name': 'resmlp',
                            'mparam': {'L': L,
                                        'T': T,
                                        'feature_sizes': (168,),
                                        }
                            }
                    }
        # ----------------------------------------------------------------------------------------------------------------------

    if x == 2:
        # conv experiments -----------------------------------------------------------------------------------------------------
        expt_name = f'conv_{seed}'
        mparam_dict = {'all': {'model_name': 'conv',
                            'mparam': {'L': L,
                                        'T': T,
                                        'channels': (5,),
                                        'kernel_sizes': (6,),
                                        'output_kernel_size': 12
                                        }
                            }
                    }
        # ----------------------------------------------------------------------------------------------------------------------

    expt_name =  os.path.join('analysis',expt_name)

    building_indices = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]#0,3,9,11,12,15,16,25,26,32,38,44,45,48,49

    start = time.time()
    predictor = Predictor(mparam_dict, building_indices, L, T, expt_name, load=False)
    predictor.train(patience=100, max_epoch=500)
    end = time.time()

    print("Train time:", end-start)

    with open(os.path.join('models','dms','resources',expt_name,'training_time.csv'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Create & train time (s)', end-start])

    # clear learning rate checkpoints from current directory
    lr_checkpoint_list = glob.glob('.lr_find*')
    for lr_checkpoint in lr_checkpoint_list:
        os.remove(lr_checkpoint)
