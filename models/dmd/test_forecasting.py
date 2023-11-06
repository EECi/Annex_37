"""Test DMD forecasting and plot."""

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
from models import DMDPredictor

def forecast(predictor, schema_path, tau, T, **kwargs):

    print("Starting forecasting.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Predictor object.
    predictor.initialise_forecasting(env)

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
            forecast_kwargs['t'] = env.time_step

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

    print("Forecasting complete.")

    return [load_logs, pv_gen_logs, pricing_logs, carbon_logs]



if __name__ == '__main__':
    import warnings

    UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings
    UCam_ids = [5,11,14,16,24,29]

    tau = 48  # model prediction horizon (number of timesteps of data predicted)
    T = 24*7*4 # number of time instances to forecast for
    dataset_dir = os.path.join('example', 'test')  # dataset directory
    schema_path = os.path.join('data', dataset_dir, 'schema.json')

    print("Initialising predictor.")
    predictor = DMDPredictor(building_indices=UCam_ids,dataset_dir=os.path.join('data','example'))

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'cvxpy')
        logs = forecast(predictor, schema_path, tau, T)

    # TODO: plot forecasts for inspection