#!/usr/bin/env python
"""Example testing file for model implementation."""

import os
import time
import numpy as np

from citylearn.citylearn import CityLearnEnv
from models import ExamplePredictor


def train_model(schema_path, tau, model_object, **kwargs):
    """Train instance of ExamplePredictor model.

    Args:
        schema_path (Str or os.Path): path to schema defining simulation data.
        tau (int): length of planning horizon
        model_object (ExamplePredictor): model instance to be trained

    Returns:
        training_stats (dict): dictionary containing useful results on how the
        training went.
    """

    print("Starting training.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Grab training data.
    ...
    ...
    ...

    print("Training complete.")

    # Save trained model.
    ...

    # Report training summary statistics.
    print("=========================Results=========================")
    ...

    # Format returned results.
    training_stats = {
        ...
    }

    return training_stats


if __name__ == '__main__':

    dataset_dir = os.path.join('example','train') # dataset directory

    schema_path = os.path.join('data',dataset_dir,'schema.json')

    tau = 48 # forecasting horizon

    # Set up model instance.
    # ========================================================================
    ... # maybe some additional setup
    predictor = ExamplePredictor(b_inds, tau)
    ... # potentially some loading
    path_to_model_file = ...
    predictor.load(path_to_model_file)
    ...

    training_stats = train_model(schema_path, tau, predictor)
