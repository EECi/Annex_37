#!/usr/bin/env python
"""
Evaluate performance of predictor model.

Apply linear MPC with provided predictor model to CityLearn environment
with specified dataset to evaluate predictor performance.
"""

import os
import time
import numpy as np
import cvxpy as cp

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from predictor import Predictor


def evaluate(schema_path, tau, **kwargs):
    print("Starting evaluation.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Linear MPC object.
    lp = LinProgModel(env=env)
    lp.set_battery_propery_data()
    lp.tau = tau
    lp.generate_LP()

    # Initialise Predictor object.

    # ========================================================================
    # insert your import & setup code for your predictor here.
    # ========================================================================

    predictor = Predictor(len(lp.b_inds), tau)

    # Initialise control loop.
    forecast_time_elapsed = 0
    lp_solver_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()
    current_socs = np.array(observations)[:,22] # get initial SoCs

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
        current_socs = np.array(observations)[:, 22]

        num_steps += 1

    print("Evaluation complete.")


    metrics = env.evaluate()    # Provides a break down of other metrics that might be of interest.
    if np.any(np.isnan(metrics['value'])):
        raise ValueError("Some of the metrics returned are NaN, please contact organizers.")

    print("=========================Results=========================")
    print(f"Price Cost: {metrics.iloc[5].value}")
    print(f"Emission Cost: {metrics.iloc[2].value}")
    print(f"Grid Cost: {np.mean([metrics.iloc[0].value, metrics.iloc[6].value])}")
    print(f"Total time taken by Predictor: {forecast_time_elapsed}s")
    print(f"Total time taken by LP solver: {lp_solver_time_elapsed}s")

    # ========================================================================
    # insert your logging code here
    # ========================================================================


if __name__ == '__main__':
    import warnings

    tau = 12    # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('example', 'test')   # dataset directory

    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        evaluate(schema_path, tau)
