"""Assess performance of forecast with explicit noise."""

import os
import warnings

from citylearn.citylearn import CityLearnEnv

from evaluate import evaluate, save_results

from models import GRWN_Predictor


if __name__ == "__main__":

    # index = int(sys.argv[1]) # for ($var = 0; $var -le 14; $var++) {python assess_forecasts.py $var}

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    # Evaluation parameters
    objective_dict = {'price':0.45,'carbon':0.45,'ramping':0.1}
    clip_level = 'b'     # aggregation level for objective

    results_file = os.path.join('results', 'prediction_tests_cntrl_sens.csv.csv')


    test_noise_levels = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    # nl = test_noise_levels[index]

    for nl in test_noise_levels:

        noise_levels = {
            'load': {'UCam_Building_%s'%id: nl for id in UCam_ids},
            'solar': {'UCam_Building_%s'%id: nl for id in UCam_ids},
            'pricing': nl,
            'carbon': nl
        }
        predictor = GRWN_Predictor(CityLearnEnv(schema=schema_path), tau, noise_levels)

        print("Assessing forecasts for noise level %s."%nl)

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module=r'cvxpy')
            results = evaluate(predictor, schema_path, tau, objective_dict, clip_level, train_building_index=None)

        results.update({
            'model_name': 'noise-all-%s'%nl,
            'tau': tau
            })

        print(results)

        save_results(results, results_file)