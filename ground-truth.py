#!/usr/bin/env python
"""
Compute performance of LinMPC controller with 'ground truth' forecasts
on given dataset.
"""

import os
import json
import time
import numpy as np
import cvxpy as cp

from tqdm import tqdm

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel


def evaluate_ground_truth(
    schema_path,
    tau,
    objective_dict = {'price':True,'carbon':True,'ramping':True},
    clip_level = 'd',
    abuff_length = 1,
    **kwargs
    ):
    """Evaluate performance of LinMPC controller using perfect ('ground truth') forecasts.

    Args:
        schema_path (str or os.Path): path to schema defining simulation data.
        tau (int): length of planning horizon
        objective_dict (dict, optional): dictionary defining objective contributions
        to use in LinMPC objective. Defaults to {'price':True,'carbon':True,'ramping':True}.
        clip_level (str, optional): 'd' (district) or 'b' (building). Level at which to clip
        power values when computing costs for objective. For 'd', grid power flows are clipped,
        meaning the costs are those for the overall portfolio of buildings, allowing energy
        transfer between building. For 'b', building power flows are clipped and the costs are
        the mean building level costs, assuming no interaction effects/cost coordination. Defaults to 'd'.
        abuff_length (int): length of control action aggregation buffer to use.

    Returns:
        results (dict): dictionary containing costs (contributions & overall) from model evaluation, and
        LP solve time.
    """

    print("Starting evaluation.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    for b in range(len(env.buildings)): # hack prices to be non-neg
        env.buildings[b].pricing.electricity_pricing = np.maximum(env.buildings[b].pricing.electricity_pricing,0)

    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    lp.set_battery_propery_data()
    lp.tau = tau
    lp.generate_LP(objective_dict=objective_dict, clip_level=clip_level)

    # Initialise control loop.
    lp_solver_time_elapsed = 0
    num_steps = 0
    done = False

    # Initialise environment.
    observations = env.reset()
    soc_obs_index = 22
    current_socs = np.array([charge*capacity for charge,capacity in zip(np.array(observations)[:,soc_obs_index],lp.battery_capacities)]) # get initial SoCs

    # Create action buffer for control action batching.
    actions_buffer = []
    assert abuff_length <= tau, "Action buffer length cannot exceed planning horizon."

    # Execute control loop.
    with tqdm(total=env.time_steps) as pbar:

        while not done:
            if num_steps%100 == 0:
                pbar.update(100)

            # Compute MPC action.
            # ====================================================================

            if len(actions_buffer) > 0: # take action from buffer
                actions = actions_buffer[0]
                actions_buffer.pop(0)

            else: # compute an action
                if num_steps <= (env.time_steps - 1) - tau:
                    # setup and solve predictive Linear Program model of system
                    lp_start = time.perf_counter()
                    lp.set_time_data_from_env(t_start=num_steps, tau=tau, current_socs=current_socs) # load ground truth data
                    lp.set_LP_parameters()
                    _,_,_,_,alpha_star = lp.solve_LP()
                    actions: np.array = alpha_star[:,0].reshape(len(lp.b_inds),1)
                    lp_solver_time_elapsed += time.perf_counter() - lp_start

                    actions_buffer = [alpha_star[:,t].reshape(len(lp.b_inds),1) for t in range(1,abuff_length)] if abuff_length > 1 else []

                else: # if not enough time left to grab a full length ground truth forecast: do nothing
                    actions = np.zeros((len(lp.b_inds),1))

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

    if clip_level == 'd': # manually compute old metrics
        price_cost = np.sum(np.maximum(env.net_electricity_consumption_price,0))/np.sum(np.maximum(env.net_electricity_consumption_without_storage_price,0))
        emissions_cost = np.sum(np.maximum(env.net_electricity_consumption_emission,0))/np.sum(np.maximum(env.net_electricity_consumption_without_storage_emission,0))

    elif clip_level == 'b': # get new metrics from env evaluation method
        price_cost = metrics.iloc[5].value if objective_dict['price'] else np.NaN
        emissions_cost = metrics.iloc[2].value if objective_dict['carbon'] else np.NaN

    grid_cost = np.mean([metrics.iloc[0].value, metrics.iloc[6].value]) if objective_dict['ramping'] else np.NaN
    cost_contributions = np.array([price_cost,emissions_cost,grid_cost])
    cost_weights = [objective_dict[key] for key in ['price', 'carbon', 'ramping']]    # enforce ordering
    if True in cost_weights: cost_weights = np.array([1/cost_weights.count(True) if item == True else 0 for item in cost_weights])
    overall_cost = cost_contributions[~np.isnan(cost_contributions)] @ cost_weights[~np.isnan(cost_contributions)]

    print("=========================Results=========================")
    print(f"Price Cost: {round(price_cost,5)}")
    print(f"Emissions Cost: {round(emissions_cost,5)}")
    print(f"Grid Cost: {round(grid_cost,5)}")
    print(f"Overall Cost: {round(overall_cost,5)}")
    print(f"Total time taken by LP solver: {round(lp_solver_time_elapsed,1)}s")

    results = {
        'Price Cost':price_cost,
        'Emissions Cost':emissions_cost,
        'Grid Cost':grid_cost,
        'Overall Cost':overall_cost,
        'Solve Time':lp_solver_time_elapsed
    }

    return results



if __name__ == '__main__':
    import warnings

    dataset_dir = os.path.join('example','test') # dataset directory

    schema_path = os.path.join('data',dataset_dir,'schema.json')

    save_path = os.path.join('results','ground_truth_evaluations-%s.json')
    ground_truth_results = {}

    objective_dict = {'price':True,'carbon':True,'ramping':True}
    clip_level = 'b' # aggregation level for objective

    taus = [6,12,24,48,72,120,168] # model prediction horizon (number of timesteps of data predicted)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',module=r'cvxpy')

        for tau in taus:
            abuff_length = 1
            print("=======================Evaluation========================")
            print(f"Tau: {tau}")
            results = evaluate_ground_truth(schema_path, tau, objective_dict, clip_level, abuff_length)
            print("\n")

            results['Tau'] = tau
            ground_truth_results[tau] = results

    objective_dict['clip_level'] = clip_level
    ground_truth_results['objective'] = objective_dict

    with open(save_path%clip_level, 'w') as json_file:
        json.dump(ground_truth_results, json_file, indent=4)
