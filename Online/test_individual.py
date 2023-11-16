from models.dms.predictor import Predictor
from models.dms.utils import IndividualPlotter
import os
building_index = 0
dataset_type = 'load'
model='linear_1'
expt_name = os.path.join('analysis',model)
predictor = Predictor(expt_name=expt_name, building_indices= [building_index],load=True) # add [] on the column CQ
x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

plotter = IndividualPlotter(x, pred, dataset_type, window_size=500)
plotter.show()
