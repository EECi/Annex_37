#!/usr/bin/env python
"""
Evaluate performance of predictor model.

Apply linear MPC with provided predictor model to CityLearn evironment
with specified dataset to evaluate predictor performance.
"""

import os
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from linmodel import LinProgModel
from predictor import Predictor


def evaluate(schema_path, tau, **kwargs):
    print("Starting evaluation.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Linear MPC object.
    lp = LinProgModel(env)
    lp.set_battery_propery_data()
    LinProgModel.tau = tau
    LinProgModel.generate_LP()

    # Initialise Predictor object.

    # ========================================================================
    # insert your import & setup code for your predictor here.
    # ========================================================================

    predictor = Predictor(...)


    # Initialise control loop.
    agent_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()
    current_socs = np.array(observations)[:,22] # get initial SoCs

    # Execute control loop.
    while not done:

        # Compute MPC action.
        step_start = time.perf_counter()

        # make forecasts using predictor
        forecasts = predictor.compute_forecast(observations)

        # setup and solve predictive Linear Program model of system
        LinProgModel.set_custom_time_data(*forecasts, current_socs=current_socs)
        LinProgModel.set_LP_parameters()
        _,_,_,alpha_star = LinProgModel.solve_LP('SCIPY',False,scipy_options={'method':'highs'})
        actions: np.array = alpha_star[:,0].reshape(len(LinProgModel.b_inds),1)

        agent_time_elapsed += time.perf_counter() - step_start

        # ====================================================================
        # insert your logging code here
        # ====================================================================

        # Apply action to environment.
        observations, _, done, _ = env.step(actions)

        num_steps += 1
        if num_steps % 1000 == 0:
            print(f"Num Steps: {num_steps}")

    print("Evaluation complete.")


    metrics = env.evaluate()  # Provides a break down of other metrics that might be of interest.
    if np.any(np.isnan(metrics['value'])):
        raise ValueError("Some of the metrics returned are NaN, please contant organizers.")

    print("=========================Results=========================")
    print(f"Price Cost: {metrics.iloc[5].value}")
    print(f"Emission Cost: {metrics.iloc[2].value}")
    print(f"Grid Cost:{np.mean([metrics.iloc[0].value, metrics.iloc[6].value])}")
    print(f"Total time taken by agent: {agent_time_elapsed}s")

    # ========================================================================
    # insert your logging code here
    # ========================================================================



if __name__ == '__main__':
    # todo: remove the citylearn_challenge_2022_phase_1 dataset from the data folder
    # todo: add the new test dataset to the data folder
    # todo: change the schema path to the test dataset.

    tau = 48 # model prediction horizon (number of timesteps of data predicted)
    dataset_dir = os.path.join('citylearn_challenge_2022_phase_1') # dataset directory

    schema_path = os.path.join('data',dataset_dir,'schema.json')

    evaluate(schema_path, tau)
