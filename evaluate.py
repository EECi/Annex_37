#!/usr/bin/env python
"""
Evaluate performance of predictor model.

Apply linear MPC with provided predictor model to CityLearn environment
with specified dataset to evaluate predictor performance.
"""

import os
import sys
import csv
import time
import numpy as np
import cvxpy as cp

import itertools

from tqdm import tqdm

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from models import (
    ExamplePredictor,
    DMSPredictor,
    TFT_Predictor,
    NHiTS_Predictor,
    DeepAR_Predictor,
    LSTM_Predictor,
    GRU_Predictor,
    GRWN_Predictor,
    LGBMOnlinePredictor_pre_trained
)


def evaluate(predictor,
             schema_path,
             tau,
             objective_dict={'price': True, 'carbon': True, 'ramping': True},
             clip_level='d',
             **kwargs):
    """Evaluate performance of LinMPC controller with given Predictor model.

    Args:
        predictor (Predictor): instantiated predictor class that inherits from models.BasePredictorModel.
        schema_path (Str or os.Path): path to schema defining simulation data.
        tau (int): length of planning horizon
        objective_dict (dict, optional): dictionary defining objective contributions
        to use in LinMPC objective. Defaults to {'price':True,'carbon':True,'ramping':True}.
        clip_level (str, optional): 'd' (district) or 'b' (building). Level at which to clip
        power values when computing costs for objective. For 'd', grid power flows are clipped,
        meaning the costs are those for the overall portfolio of buildings, allowing energy
        transfer between building. For 'b', building power flows are clipped and the costs are
        the mean building level costs, assuming no interaction effects/cost coordination. Defaults to 'd'.

    Returns:
        results (dict): dictionary containing costs (contributions & overall) from model evaluation, and
        forecasting & LP solve time.
    """

    print("Starting evaluation.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Predictor object.
    if type(predictor) in [ExamplePredictor, TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor, LSTM_Predictor, GRU_Predictor]:
        predictor.initialise_forecasting(tau, env)
    elif type(predictor) in [DMSPredictor, LGBMOnlinePredictor_pre_trained]:
        predictor.initialise_forecasting(env)
        
    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    lp.set_battery_propery_data()
    lp.tau = tau
    lp.generate_LP(objective_dict=objective_dict, clip_level=clip_level)

    # Initialise control loop.
    forecast_time_elapsed = 0
    lp_solver_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()
    soc_obs_index = 22
    current_socs = np.array([charge*capacity for charge,capacity in zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities)]) # get initial SoCs

    # Execute control loop.
    with tqdm(total=env.time_steps) as pbar:
        while not done:
            if num_steps % 100 == 0:
                pbar.update(100)

            # Compute MPC action.
            # ====================================================================

            # Set up custom data input for method.
            forecast_kwargs = {}
            if type(predictor) in [TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor, LSTM_Predictor, GRU_Predictor, GRWN_Predictor, LGBMOnlinePredictor_pre_trained]:
                forecast_kwargs['t'] = env.time_step
            elif type(predictor) in [DMSPredictor]:
                forecast_kwargs['train_building_index'] = kwargs['train_building_index']

            # make forecasts using predictor
            forecast_start = time.perf_counter()
            forecasts = predictor.compute_forecast(observations, **forecast_kwargs)
            forecast_time_elapsed += time.perf_counter() - forecast_start

            if forecasts is None:   # forecastor opt out - ##  or ((env.time_steps - 1) - env.time_step < tau)
                actions = np.zeros((len(lp.b_inds), 1))
            else:
                forecasts = list(forecasts)
                forecasts[0] = forecasts[0].reshape(len(env.buildings), -1)
                forecasts[1] = np.array([b.pv.get_generation(forecasts[1]) for b in env.buildings])
                forecasts[2] = forecasts[2].reshape(-1)
                forecasts[3] = forecasts[3].reshape(-1)
                # ================================================================

                # setup and solve predictive Linear Program model of system
                lp_start = time.perf_counter()
                lp.set_custom_time_data(*forecasts, current_socs=current_socs)
                lp.set_LP_parameters()
                _, _, _, _, alpha_star = lp.solve_LP()
                actions: np.array = alpha_star[:, 0].reshape(len(lp.b_inds), 1)
                lp_solver_time_elapsed += time.perf_counter() - lp_start

            # Apply action to environment.
            # ====================================================================
            observations, _, done, _ = env.step(actions)

            # Update battery states-of-charge
            # ====================================================================
            current_socs = np.array([charge*capacity for charge,capacity in zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities)])

            num_steps += 1

    print("Evaluation complete.")

    metrics = env.evaluate()  # provides a break down of other metrics that might be of interest.
    if np.any(np.isnan(metrics['value'])):
        raise ValueError("Some of the metrics returned are NaN, please contact organizers.")

    if clip_level == 'd':   # manually compute old metrics
        price_cost = np.sum(np.maximum(env.net_electricity_consumption_price,0))/np.sum(np.maximum(env.net_electricity_consumption_without_storage_price,0))
        emissions_cost = np.sum(np.maximum(env.net_electricity_consumption_emission,0))/np.sum(np.maximum(env.net_electricity_consumption_without_storage_emission,0))

    elif clip_level == 'b':     # get new metrics from env evaluation method
        price_cost = metrics.iloc[5].value if objective_dict['price'] else np.NaN
        emissions_cost = metrics.iloc[2].value if objective_dict['carbon'] else np.NaN

    grid_cost = np.mean([metrics.iloc[0].value, metrics.iloc[6].value]) if objective_dict['ramping'] else np.NaN
    cost_contributions = np.array([price_cost, emissions_cost, grid_cost])
    cost_weights = np.array([objective_dict[key] for key in ['price', 'carbon', 'ramping']])    # enforce ordering
    if True in cost_weights: cost_weights = np.array([1/cost_weights.count(True) if item == True else 0 for item in cost_weights])
    overall_cost = cost_contributions[~np.isnan(cost_contributions)] @ cost_weights[~np.isnan(cost_contributions)]

    print("=========================Results=========================")
    print(f"Price Cost: {round(price_cost, 5)}")
    print(f"Emissions Cost: {round(emissions_cost, 5)}")
    print(f"Grid Cost: {round(grid_cost, 5)}")
    print(f"Overall Cost: {round(overall_cost, 5)}")
    print(f"Total time taken by LP solver: {round(lp_solver_time_elapsed, 1)}s")

    results = {
        'Price Cost': price_cost,
        'Emissions Cost': emissions_cost,
        'Grid Cost': grid_cost,
        'Overall Cost': overall_cost,
        'Forecast Time': forecast_time_elapsed,
        'Solve Time': lp_solver_time_elapsed
    }

    # ========================================================================
    # insert your logging code here
    # ========================================================================

    return results


def save_results(results, results_file):
    """Save control performance results to CSV.
    
    Args:
        results (dict): dictionary containing control performance results.
        results_file (str or os.Path): path to CSV file results are saved to.
    """

    header = ['Model', 'Forecast Time (s)', 'Solve Time (s)', 'Tau (hrs)', 'Overall', 'Price', 'Carbon', 'Grid']
    out = [results['model_name'], results['Forecast Time'], results['Solve Time'], results['tau'], results['Overall Cost'], results['Price Cost'], results['Emissions Cost'], results['Grid Cost']]

    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
    with open(results_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(out)


if __name__ == '__main__':
    import warnings

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings

    # index = int(sys.argv[1]) # for ($var = 0; $var -le 14; $var++) {python assess_forecasts.py $var}
    # b_id = UCam_ids[index]

    # Set parameters and instantiate predictor
    # ==================================================================================================================
    dataset_dir_for_predictor = os.path.join('data', 'analysis')  # dataset directory

    dataset_dir = os.path.join('analysis', 'test')   # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')
    tau = 48    # model prediction horizon (number of timesteps of data predicted)

    # Evaluation parameters
    objective_dict = {'price':0.45,'carbon':0.45,'ramping':0.1}
    clip_level = 'b'     # aggregation level for objective
    # TODO: add mixed objective clip level option

    save = True
    # model_name = os.path.join('analysis','linear_0')
    model_name = os.path.join('analysis','LGBM')

    train_building_index = None
    results_file = os.path.join('results', 'evaluate_tests_same-train-test--temp.csv')

    # Instantiate Predictor
    # predictor = DMSPredictor(building_indices=UCam_ids, expt_name=model_name, load=True)
    # predictor = TFT_Predictor(model_group_name='analysis')
    # predictor = LGBMOnlinePredictor(building_indices=UCam_ids, dataset_dir = dataset_dir_for_predictor, lags=list(itertools.chain(range(1, 49), range(72, 721, 24))))
    # predictor = SARIMAXPredictor(building_indices=UCam_ids, dataset_dir = dataset_dir_for_predictor, use_seasonality=True)
    # predictor = XGBPreTPredictor(len(UCam_ids), tau, "_H1") #(len(lp.b_inds), tau, "_H1")
    # predictor = LGBMOnlinePredictor_iterative(building_indices=UCam_ids, dataset_dir = dataset_dir_for_predictor, lags=[])
    predictor = LGBMOnlinePredictor_pre_trained(building_indices=UCam_ids, dataset_dir = dataset_dir_for_predictor, iterative=False)

    # ==================================================================================================================
    # Evaluate and save results
    # ==================================================================================================================
    print("Evaluating controller for model %s."%model_name)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = evaluate(predictor, schema_path, tau, objective_dict, clip_level, train_building_index=train_building_index)

    results.update({
        'model_name': model_name,
        'tau': tau
        })
    print(results)

    if save: save_results(results, results_file)