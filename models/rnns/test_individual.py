from models.rnns.predictor import Predictor
from models.rnns.utils import IndividualPlotter
import matplotlib
matplotlib.use('Qt5Agg')
building_index = 5
dataset_type = 'load'
expt_name = 'lstm_L_168_T48'#'linear_L_168_T48'#'DeepAR_L_168_T48'
predictor = Predictor(expt_name=expt_name, load=True)

#predictor.models['load_5'] (predictor.buffers['load_5'])
x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')
# x.shape
# pred.shape
plotter = IndividualPlotter(x, pred, dataset_type, window_size=500)
plotter.show()
