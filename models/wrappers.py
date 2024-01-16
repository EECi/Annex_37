"""Handy wrappers for forecasting model classes."""

import numpy as np
from .dms import Predictor as DMSPredictor

class BuildingForecastsOnlyWrapper():
    """Prediction model wrapper that gets building forecasts from a DMS model
    and returns alongside zeros for other variables when forecasts are computed.
    """
    def __init__(self, predictor: DMSPredictor, tau: int):

        self.predictor = predictor
        self.tau = tau

    def compute_forecast(self, observations, train_building_index=None):

        predicted_loads,_,_,_ = self.predictor.compute_forecast(observations, train_building_index)

        return predicted_loads, np.zeros((self.tau)), np.zeros((self.tau)), np.zeros((self.tau))

    def initialise_forecasting(self, env):

        self.predictor.initialise_forecasting(env)