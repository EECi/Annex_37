from models.dms.predictor import Predictor
from models.dms.utils import IndividualPlotter

building_index = 0
dataset_type = 'price'
feature_number = 11
expt_name = f'linear_1_feature{feature_number}'
predictor = Predictor(expt_name=expt_name, load=True)
x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

plotter = IndividualPlotter(x, pred, dataset_type, window_size=500)
plotter.show()
