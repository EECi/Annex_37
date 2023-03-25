from predictor import Predictor


L = 168
T = 48
expt_name = f'linear_L{L}_T{T}'
mparam_dict = {'all': {'model_name': 'vanilla',
                       'mparam': {'L': L,
                                  'T': T,
                                  'layers': []}}}
building_indices = (5, 11, 14, 16, 24, 29)
predictor = Predictor(mparam_dict, building_indices, L, T, expt_name)
predictor.train(patience=25, max_epoch=500)
