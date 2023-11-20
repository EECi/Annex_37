"""Assess performance of forecast with explicit noise."""

import os
import sys
import warnings

from citylearn.citylearn import CityLearnEnv

from evaluate import evaluate, save_results

from models import GRWN_Predictor


if __name__ == "__main__":

    # Run using
    # for ($v = 0; $v -le 4; $v++) {for ($nl = 0; $nl -le 14; $nl++) {python -m experiments.cntrl_sens_control $v $nl}}
    # ==================================================================================================

    noise_var_ind = int(sys.argv[1])
    index = int(sys.argv[2])

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    # Evaluation parameters
    objective_dict = {'price':0.45,'carbon':0.45,'ramping':0.1}
    clip_level = 'b'     # aggregation level for objective

    results_file = os.path.join('results', 'evaluate_tests_cntrl_sens.csv')


    noised_vars = ['all','load','solar','pricing','carbon']
    test_noise_levels = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

    noise_var = noised_vars[noise_var_ind]
    nl = test_noise_levels[index]

    noise_levels = {
        'load': {'UCam_Building_%s'%id: nl for id in UCam_ids},
        'solar': nl,
        'pricing': nl,
        'carbon': nl
    }

    if noise_var != 'all': # set all other noise levels expect for selected variable to 0
        for key in noise_levels.keys():
            if key != noise_var:
                if key == 'load':
                    noise_levels[key] = {'UCam_Building_%s'%id: 0 for id in UCam_ids}
                else:
                    noise_levels[key] = 0
    else: # keep noise levels on all variables
        pass

    predictor = GRWN_Predictor(CityLearnEnv(schema=schema_path), tau, noise_levels)

    print("Assessing forecasts for noise level %s-%s."%(noise_var,nl))

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = evaluate(predictor, schema_path, tau, objective_dict, clip_level, train_building_index=None)

    results.update({
        'model_name': 'noise-%s-%s'%(noise_var,nl),
        'tau': tau
        })

    print(results)

    save_results(results, results_file)