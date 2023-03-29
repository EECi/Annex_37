from predictor import Predictor
from utils.dms import IndividualInference


building_index = 5
dataset_type = 'carbon'

expt_name = 'linear_L168_T168'
predictor = Predictor(expt_name=expt_name, load=True)
_, pred, gt, gt_t, pred_t, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

inference = IndividualInference(pred, gt, gt_t, pred_t)
inference.show()
