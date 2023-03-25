from predictor import Predictor
from utils.pat import IndividualPlotter


building_index = 5
dataset_type = 'price'

expt_name = 'linear_L168_T48'
predictor = Predictor(expt_name=expt_name, load=True)
x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

plotter = IndividualPlotter(x, pred, window_size=500)
plotter.show()
