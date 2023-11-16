import os
from models.dms.predictor import Predictor

building_indices = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49]

for model in ['linear_0','resmlp_0','conv_0']:
    results_file = 'results.csv'
    expt_name = os.path.join('analysis',model)
    predictor = Predictor(building_indices=building_indices, expt_name=expt_name, load=True, results_file=results_file)
    header, results = predictor.test()
