import os
import glob
from models.dms.predictor import Predictor


L = 168
T = 48
expt_name = f'H256_L{L}_T{T}'
mparam_dict = {'all': {'model_name': 'vanilla',
                       'mparam': {'L': L,
                                  'T': T,
                                  'layers': [256]}}}
building_indices = (5, 11, 14, 16, 24, 29)
predictor = Predictor(mparam_dict, building_indices, L, T, expt_name, load=False)
predictor.train(patience=25, max_epoch=500)

# clear learning rate checkpoints from current directory
lr_checkpoint_list = glob.glob('.lr_find*')
for lr_checkpoint in lr_checkpoint_list:
    os.remove(lr_checkpoint)
