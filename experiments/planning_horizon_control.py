"""Assess performance of perfect forecast control (ground truth)
    with varying planning horizon."""

import os
import sys
import warnings

from citylearn.citylearn import CityLearnEnv

from evaluate import evaluate, save_results

from models import GRWN_Predictor


if __name__ == "__main__":

    # Run using
    # for ($ph = 0; $ph -le 12; $ph++) {python -m experiments.planning_horizon_control $ph}
    # ==================================================================================================

    # NOTE: evaluation fails for tau >= 96 due to ill-conditioned LPs in some time instances

    index = int(sys.argv[1])

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    dataset_dir = os.path.join('analysis', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    # Evaluation parameters
    objective_dict = {'price':0.45,'carbon':0.45,'ramping':0.1}
    clip_level = 'b'     # aggregation level for objective

    results_file = os.path.join('results', 'evaluate_tests_planning_horizon.csv')


    planning_horizons = [2,3,6,9,12,18,24,32,48,72,96,120,168] # can't do tau=1 due to ramp objective
    tau = planning_horizons[index]  # model prediction horizon (number of timesteps of data predicted)

    nl = 0 # no noise - use perfect forecasts
    noise_levels = {
        'load': {'UCam_Building_%s'%id: nl for id in UCam_ids},
        'solar': nl,
        'pricing': nl,
        'carbon': nl
    }

    predictor = GRWN_Predictor(CityLearnEnv(schema=schema_path), tau, noise_levels)

    print("Assessing forecasts for planning horizon, tau = %s."%tau)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = evaluate(predictor, schema_path, tau, objective_dict, clip_level, train_building_index=None)

    results.update({
        'model_name': 'no-noise-tau-%s'%tau,
        'tau': tau
        })

    print(results)

    save_results(results, results_file)