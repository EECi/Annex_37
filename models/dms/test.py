from models.dms.predictor import Predictor

results_file = 'results.csv'
expt_name = 'C5K6O6P6'
predictor = Predictor(expt_name=expt_name, load=True, results_file=results_file)
header, results = predictor.test()
