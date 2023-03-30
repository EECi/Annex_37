import os
import glob
from predictor import Predictor


L = 48
T = 48
expt_name = f'linear_L{L}_T{T}'
mparam_dict = {'all': {'model_name': 'vanilla',
                       'mparam': {'L': L,
                                  'T': T,
                                  'layers': []}}}
building_indices = (5, 11, 14, 16, 24, 29)
predictor = Predictor(mparam_dict, building_indices, L, T, expt_name)
# predictor.train(patience=25, max_epoch=500)
predictor.train(patience=5, max_epoch=10)

# clear learning rate checkpoints from current directory
lr_checkpoint_list = glob.glob('.lr_find*')
for lr_checkpoint in lr_checkpoint_list:
    os.remove(lr_checkpoint)
