from models.dms.predictor import Predictor

results_file = 'results.csv'
expt_name = 'H256_L168_T48'
predictor = Predictor(expt_name=expt_name, load=True, results_file=results_file)
header, results = predictor.test()
