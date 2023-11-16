from models.dms.predictor import Predictor
from models.dms.utils import IndividualInference

building_index = 5
dataset_type = 'carbon'

expt_name = 't_d128_l4_h16_p1'
predictor = Predictor(expt_name=expt_name, load=True)

_, pred, gt, gt_t, pred_t, mse = predictor.test_individual(building_index, dataset_type)
print(f'mse = {mse}')

inference = IndividualInference(pred, gt, gt_t, pred_t, dataset_type)
inference.show()
