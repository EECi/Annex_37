#!/usr/bin/env python
"""
Assess performance of predictor model forecasts.

Perform prediction inference using given predictor model with
specified dataset to evaluate predictor forecasting performance
in comparison to ground truth values of prediction variables.
"""

import os
import sys
import csv
import time
import numpy as np

from tqdm import tqdm

from citylearn.citylearn import CityLearnEnv
from models import (
    DMSPredictor,
    TFT_Predictor,
    NHiTS_Predictor,
    DeepAR_Predictor,
    LSTM_Predictor,
    GRU_Predictor,
    GRWN_Predictor
)


def compute_metric_score(forecasts_array, ground_truth_array, metric, global_mean_norm=True):
    """Compute mean metric score over set of forecasts corresponding
    to ground truth arrays for specified metric function.

    Args:
        forecasts_array (List[List]): list of forecast list to compute metric values for.
        ground_truth_array (List[List]): list of ground truth value lists corresponding to
        forecasts (length can be less than or equal to corresponding forecast length).
        metric (function): function computing desired forecast performance metric
        for a given forecast and corresponding ground truth (taken as `np.array`s).
        mean_norm (bool, optional): Whether to normalise the mean metric value by the
        mean of the underlying ground truth timeseries. Defaults to False.

    Returns:
        metric_score (float): mean metric score over set of forecasts & ground truths.
    """

    assert len(forecasts_array) == len(ground_truth_array), "Must provide same number of forecasts and ground truths to compare."

    metric_scores = []
    for forecast, actual in zip(forecasts_array, ground_truth_array):
        a = np.array(actual)
        f = np.array(forecast)[:len(a)]
        metric_scores.append(metric(f, a))

    metric_score = np.mean(metric_scores)

    if global_mean_norm:
        metric_score = metric_score/np.mean([l[0] for l in ground_truth_array if len(l) > 0])

    return metric_score


def MAE(prediction, actual):
    return np.mean(np.abs((prediction-actual)))


def RMSE(prediction, actual):
    return np.sqrt(np.mean(np.power(prediction-actual, 2)))


def assess(predictor, schema_path, tau, building_breakdown=False, **kwargs):
    """Evaluate forecasting performance of given Predictor model for
    dataset specified by provided schema.

    Args:
        predictor (Predictor): instantiated predictor class that inherits from models.BasePredictorModel.
        schema_path (Str or os.Path): path to schema defining simulation data.
        tau (int): length of planning horizon
        building_breakdown (bool): indicator for whether building resolved
        performance metric values are reported. Defaults to 'False'.

    Returns:
        results (dict): dictionary containing performance metrics from forecasting
        assessment, and forecasting time.
    """

    print("Starting assessment.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Predictor object.
    if type(predictor) in [TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor, LSTM_Predictor, GRU_Predictor]:
        predictor.initialise_forecasting(tau, env)
    elif type(predictor) in [DMSPredictor]:
        predictor.initialise_forecasting(env)

    # ========================================================================
    # insert your import & setup code for your predictor here.
    # ========================================================================

    # Initialise logging objects.
    load_logs = {b.name:{'forecasts': [], 'actuals': []} for b in env.buildings}
    pv_gen_logs = {'forecasts': [], 'actuals': []}
    pricing_logs = {'forecasts': [], 'actuals': []}
    carbon_logs = {'forecasts': [], 'actuals': []}

    # Initialise forecasting loop.
    forecast_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()

    # Execute control loop.
    with tqdm(total=env.time_steps) as pbar:

        while not done:
            if num_steps % 100 == 0:
                pbar.update(100)
            # Set up custom data input for method.
            forecast_kwargs = {}
            if type(predictor) in [TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor, LSTM_Predictor, GRU_Predictor, GRWN_Predictor]:
                forecast_kwargs['t'] = env.time_step
            elif type(predictor) in [DMSPredictor]:
                forecast_kwargs['train_building_index'] = kwargs['train_building_index']

            # Compute forecast.
            forecast_start = time.perf_counter()
            forecasts = predictor.compute_forecast(observations, **forecast_kwargs)
            forecast_time_elapsed += time.perf_counter() - forecast_start
            # Perform logging.
            if forecasts is None:   # forecastor opt out
                pass    # no forecast to evaluate
            else:
                # Log forecasts.
                for i, b in enumerate(env.buildings):
                    load_logs[b.name]['forecasts'].append(forecasts[0][i].reshape(-1))
                pv_gen_logs['forecasts'].append(forecasts[1].reshape(-1))
                pricing_logs['forecasts'].append(forecasts[2].reshape(-1))
                carbon_logs['forecasts'].append(forecasts[3].reshape(-1))
                # Log ground-truth values.
                # note abuse of Python array slicing to give variable length actuals toward end of lists
                for i,b in enumerate(env.buildings):
                    load_logs[b.name]['actuals'].append(b.energy_simulation.non_shiftable_load[env.time_step+1:env.time_step+1+tau])
                pv_gen_logs['actuals'].append(b.energy_simulation.solar_generation[env.time_step+1:env.time_step+1+tau])
                pricing_logs['actuals'].append(b.pricing.electricity_pricing[env.time_step+1:env.time_step+1+tau])
                carbon_logs['actuals'].append(b.carbon_intensity.carbon_intensity[env.time_step+1:env.time_step+1+tau])

            # Step environment.
            actions = np.zeros((len(env.buildings), 1))
            observations, _, done, _ = env.step(actions)

            num_steps += 1
            #print(num_steps)

    print("Assessment complete.")

    # Compute forecasting performance metrics.
    metrics = [MAE,RMSE]
    metric_names = ['gmnMAE', 'gmnRMSE']
    globally_mean_normalised = [True, True]

    load_metrics = {
        b.name:{
            mname: compute_metric_score(load_logs[b.name]['forecasts'],load_logs[b.name]['actuals'], metric, gnorm)\
                for metric, mname, gnorm in zip(metrics, metric_names, globally_mean_normalised)
        } for b in env.buildings
    }
    load_metrics['buildings_average'] = {mname: np.mean([load_metrics[b.name][mname] for b in env.buildings])\
        for mname in metric_names}

    pv_gen_metrics = {
            mname: compute_metric_score(pv_gen_logs['forecasts'],pv_gen_logs['actuals'],metric)\
                for metric,mname in zip(metrics,metric_names)
        }

    pricing_metrics = {
            mname: compute_metric_score(pricing_logs['forecasts'],pricing_logs['actuals'],metric)\
                for metric,mname in zip(metrics,metric_names)
        }

    carbon_metrics = {
            mname: compute_metric_score(carbon_logs['forecasts'], carbon_logs['actuals'],metric)\
                for metric,mname in zip(metrics,metric_names)
        }

    print("=========================Results=========================")
    print(f"Total time taken for forecasting: {round(forecast_time_elapsed,1)}s")
    print("")
    print("=====Buildings Average=====")
    print("---Load---")
    for mname in metric_names: print(f"{mname}: {round(load_metrics['buildings_average'][mname],5)}")
    print("---Solar Generation---")
    for mname in metric_names: print(f"{mname}: {round(pv_gen_metrics[mname],5)}")
    print("=====Pricing=====")
    for mname in metric_names: print(f"{mname}: {round(pricing_metrics[mname],5)}")
    print("=====Carbon Intensity=====")
    for mname in metric_names: print(f"{mname}: {round(carbon_metrics[mname],5)}")
    print("")
    if building_breakdown:
        for b in env.buildings:
            print(f"====={b.name}=====")
            print("---Load---")
            for mname in metric_names: print(f"{mname}: {round(load_metrics[b.name][mname],5)}")

    results = {
        'Load Forecasts': load_metrics,
        'Solar Potential Forecasts': pv_gen_metrics,
        'Pricing Forecasts': pricing_metrics,
        'Carbon Intensity Forecasts': carbon_metrics,
        'Forecast Time': forecast_time_elapsed
    }
    return results



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    # index = int(sys.argv[1]) # for ($var = 0; $var -le 14; $var++) {python assess_forecasts.py $var}
    # b_id = UCam_ids[index]
    # model_extensions = ['rd4y','rd2y','rd1y','rd6m','rd3m']
    # me = model_extensions[index]
    # test_noise_levels = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1, 0.2]
    # nl = test_noise_levels[index]

    # Set parameters and instantiate predictor
    # ==================================================================================================================
    feature_numbers = range(1,12)
    for feature_number in feature_numbers:

        tau = 48  # model prediction horizon (number of timesteps of data predicted)
        dataset_dir = os.path.join('analysis', 'test')  # dataset directory
        schema_path = os.path.join('data', dataset_dir, 'schema.json')

        save = True
        # model_name = os.path.join('TFT') # 'linear_1' # todo: set expt name for saving accordingly

        model_name = os.path.join(f'analysis/linear_1_f{feature_number}N')

        train_building_index = None # int or None - b_id
        # results_file = os.path.join('results', 'prediction_tests_cntrl_sens.csv') # todo from pat: put back I guess
        results_file = os.path.join('results', 'test_normalise.csv')

        # Instantiate predictor
        predictor = DMSPredictor(building_indices=UCam_ids, expt_name=model_name, load=True,feature_number=feature_number)
        # predictor = TFT_Predictor(model_group_name='analysis') # ,model_names=[b_id]*len(UCam_ids)
        # noise_levels = {
        #     'load': {'UCam_Building_%s'%id: nl for id in UCam_ids},
        #     'solar': {'UCam_Building_%s'%id: nl for id in UCam_ids},
        #     'pricing': nl,
        #     'carbon': nl
        # }
        # predictor = GRWN_Predictor(CityLearnEnv(schema=schema_path),tau,noise_levels)


        # ==================================================================================================================
        # Assess forecasts and save results
        # ==================================================================================================================
        print("Assessing forecasts for model %s."%model_name)

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module=r'cvxpy')
            results = assess(predictor, schema_path, tau, building_breakdown=True, train_building_index=train_building_index)


        if save:

            header = ['Model Name', 'Train Building', 'Forecast Time (s)', 'Tau (hrs)', 'Metric', 'P', 'C', 'S']
            header += ['L'+i.split('_')[-1] for i in results['Load Forecasts'].keys() if 'average' not in i]

            if not os.path.exists(results_file):
                with open(results_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)

            for metric in ['gmnMAE','gmnRMSE']:
                out = [
                    model_name,
                    train_building_index if train_building_index is not None else 'same-train-test',
                    results['Forecast Time'],
                    tau,
                    metric,
                    results['Pricing Forecasts'][metric],
                    results['Carbon Intensity Forecasts'][metric],
                    results['Solar Potential Forecasts'][metric]
                ]
                for k, v in results['Load Forecasts'].items():
                    if 'average' not in k:
                        out.append(v[metric])

                with open(results_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(out)