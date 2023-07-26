#!/usr/bin/env python
"""
Evaluate performance of predictor model.

Apply linear MPC with provided predictor model to CityLearn environment
with specified dataset to evaluate predictor performance.
"""

import os
import csv
import time
import numpy as np
import cvxpy as cp

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from models import ExamplePredictor, DMSPredictor#, TFTPredictor   # todo put back


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
    while not done:
        if num_steps % 100 == 0:
            print(f"Num Steps: {num_steps} ({np.round(100*num_steps/env.time_steps,1)}%)")

        # Compute MPC action.
        # ====================================================================

        # make forecasts using predictor
        forecast_start = time.perf_counter()
        forecasts = predictor.compute_forecast(observations)
        forecast_time_elapsed += time.perf_counter() - forecast_start

        if forecasts is None:   # forecastor opt out
            actions = np.zeros((len(lp.b_inds), 1))
        else:
            forecasts = list(forecasts)
            forecasts[0] = forecasts[0].reshape(len(env.buildings), -1)
            forecasts[1] = forecasts[1].reshape(len(env.buildings), -1)
            forecasts[2] = forecasts[2].reshape(-1)
            forecasts[3] = forecasts[3].reshape(-1)

            # setup and solve predictive Linear Program model of system
            lp_start = time.perf_counter()
            lp.set_custom_time_data(*forecasts, current_socs=current_socs)
            lp.set_LP_parameters()
            _, _, _, _, alpha_star = lp.solve_LP()
            actions: np.array = alpha_star[:, 0].reshape(len(lp.b_inds), 1)
            lp_solver_time_elapsed += time.perf_counter() - lp_start

        # ====================================================================
        # insert your logging code here
        # ====================================================================

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
    cost_weights = [objective_dict[key] for key in ['price', 'carbon', 'ramping']]    # enforce ordering
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


if __name__ == '__main__':
    import warnings

    # Set parameters and instantiate predictor
    # ==================================================================================================================
    # Parameters
    save = True
    model_name = 't_d128_l4_h16_p1'
    results_file = 'evaluate_results.csv'
    results_file = os.path.join('archive_ignore/outputs', results_file)

    # Instantiate Predictor
    # predictor = ExamplePredictor(6, 48)
    predictor = DMSPredictor(expt_name=model_name, load=True)

    # Evaluation parameters
    objective_dict = {'price': True, 'carbon': True, 'ramping': True}
    clip_level = 'b'     # aggregation level for objective
    # ==================================================================================================================

    # evaluate predictor
    dataset_dir = os.path.join('example', 'test')   # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')
    tau = 48    # model prediction horizon (number of timesteps of data predicted)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        results = evaluate(predictor, schema_path, tau, objective_dict, clip_level)

    if save:
        header = ['Model', 'Overall', 'Price', 'Carbon', 'Grid']
        out = [model_name, results['Overall Cost'], results['Price Cost'], results['Emissions Cost'], results['Grid Cost']]

        if not os.path.exists(results_file):
            with open(results_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
        with open(results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(out)
