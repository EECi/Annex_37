from models.dms.predictor import Predictor
from models.dms.utils import IndividualPlotter

building_index = 3
dataset_type = 'load'

expt_name = 'linear_1'
predictor = Predictor(expt_name=expt_name, load=True)
x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

plotter = IndividualPlotter(x, pred, dataset_type, window_size=500)
plotter.show()
