import os
import glob
from models.dms.predictor import Predictor
import torch
import numpy as np
import random

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


L = 168
T = 48
expt_name = f't_d128_l4_h16_p1'
mparam_dict = {'all': {'model_name': 'transformer',
                       'mparam': {'L': L,
                                  'T': T,
                                  'hidden_dim': 128,
                                  'num_layers': 4,
                                  'num_heads': 16,
                                  'patch_size': 6,
                                  'patch_stride': 1,
                                  'dropout': 0.1
                                  }
                       }
               }

building_indices = (5, 11, 14, 16, 24, 29)
predictor = Predictor(mparam_dict, building_indices, L, T, expt_name, load=False)
predictor.train(patience=100, max_epoch=500)

# clear learning rate checkpoints from current directory
lr_checkpoint_list = glob.glob('.lr_find*')
for lr_checkpoint in lr_checkpoint_list:
    os.remove(lr_checkpoint)
