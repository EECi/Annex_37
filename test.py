from models.rnns.predictor import Predictor

results_file = 'results.csv'
expt_name = '2linear_L168_T48'
predictor = Predictor(expt_name=expt_name, load=True, results_file=results_file)
header, results = predictor.test()
