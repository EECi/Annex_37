#!/usr/bin/env python
"""Example testing file for model implementation."""

import os
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from models import ExamplePredictor


def test_forecasting(schema_path, tau, **kwargs):
    """Test performance of ExamplePredictor model implementation.

    Args:
        schema_path (Str or os.Path): path to schema defining simulation data.
        tau (int): length of planning horizon

    Returns:
        results (dict): dictionary containing useful results to understand
        how the model is performing.
    """

    print("Starting testing.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Predictor object.
    ... # maybe some additional setup
    predictor = ExamplePredictor(b_inds, tau)
    ... # potentially some loading, perhaps via the kwargs

    # Initialise forecasting loop.
    forecast_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()

    # Execute forecasting loop.
    while not done:
        if num_steps % 100 == 0:
            print(f"Num Steps: {num_steps} ({np.round(100*num_steps/env.time_steps,1)}%)")

        # Perform forecasting.
        # ====================================================================

        # make forecasts using predictor
        forecast_start = time.perf_counter()
        forecasts = predictor.compute_forecast(observations)
        forecast_time_elapsed += time.perf_counter() - forecast_start

        if forecasts is None: # forecastor opt out
            ...
            pass
        else:
            # setup and solve predictive Linear Program model of system
            ... # analyse the forecast
            ...
            ...

        # Apply dummy action to environment.
        # ====================================================================
        observations, _, done, _ = env.step(np.zeros(b_inds,1))

        num_steps += 1

    print("Testing complete.")

    # Analyse forecasting results.
    # ========================================================================
    ...

    # Report results.
    print("=========================Results=========================")
    ...

    # Format returned results.
    results = {
        ...
    }

    return results


if __name__ == '__main__':

    dataset_dir = os.path.join('example','validate') # dataset directory

    schema_path = os.path.join('data',dataset_dir,'schema.json')

    tau = 48 # forecasting horizon

    results = test_forecasting(schema_path, tau)
