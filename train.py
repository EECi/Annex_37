import os
import glob
from models.rnns.predictor import Predictor
import lightning.pytorch as pl

L = 168
T = 48
expt_name = f'DeepAR_L{L}_T{T}'
mparam_dict = {'all': {'model_name': 'DeepAR',
                       'mparam': {'L': L,
                                  'T': T,
                                  }}}

building_indices = (5, 11, 14, 16, 24, 29)
predictor = Predictor(mparam_dict, building_indices, L, T, expt_name, load=False)

# predictor.models['load_5']
predictor.train(patience=5, max_epoch=10)

# clear learning rate checkpoints from current directory
lr_checkpoint_list = glob.glob('.lr_find*')
for lr_checkpoint in lr_checkpoint_list:
    os.remove(lr_checkpoint)
