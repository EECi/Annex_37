"""
Implementation of your prediction method.

The Predictor component of the Linear MPC controller is implemented
as a class.
This class must have the following methods:
    - __init__(self, ...), which initialises the Predictor object and
        performs any initial setup you might want to do.
    - compute_forecast(observation), which executes your prediction method,
        creating timeseries forecasts for [building electrical loads,
        building solar pv generaiton powers, grid electricity price, grid
        carbon intensity] given the current observation.
"""

class Predictor:

    def __init__(self, N: int, tau: int):
        """Initialise Prediction object and perform setup."""

        self.num_buildings = N
        self.tau = tau # length of planning horizon

        # You may want to track some variables, eg. observations from previous time steps.
        # ==============================================================================================================
        self.prev_observation = None
        self.buffer = {'key': []}
        # ==============================================================================================================

    def compute_forecast(observation):
        """Compute forecasts given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation.
                The observation is a list of observations for each building (sub-list),
                where the sub-lists contain values as specified in the ReadMe.md

        Returns:
            predicted_loads (np.array): predicted electrical loads of buildings in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pv_gens (np.array): predicted energy generations of pv panels in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pricing (np.array): predicted grid electricity price in each period
                of the planning horizon ($/kWh) - shape (tau)
            predicted_carbon (np.array): predicted grid electricity carbon intensity in each
                period of the planning horizon (kgCO2/kWh) - shape (tau)
        """

        return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon
