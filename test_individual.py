from utils.pat import IndividualPlotter
from predictor import Predictor


building_index = 5
dataset_type = 'solar'

expt_name = 'linear_L144_T48'
predictor = Predictor(expt_name=expt_name, load=True)
x, pred, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

plotter = IndividualPlotter(x, pred, window_size=500)
plotter.show()
