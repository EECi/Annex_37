"""
Implementation of your prediction method.

The Predictor component of the Linear MPC controller is implemented
as a class.
This class must have the following methods:
    - __init__(self, ...), which initialises the Predictor object and
        performs any initial setup you might want to do.
    - compute_forecast(observation), which executes your prediction method,
        creating timeseries forecasts for [building electrical loads,
        building solar pv generation powers, grid electricity price, grid
        carbon intensity] given the current observation.

You may wish to implement additional methods to make your model code neater.
"""

import numpy as np


class ExamplePredictor:

    def __init__(self, N: int, tau: int):
        """Initialise Prediction object and perform setup.
        
        Args:
            N (int): number of buildings in model, hence number of buildings
                requiring forecasts.
            tau (int): length of planning horizon (number of time instances
                into the future to forecast).
                Note: with some adjustment of the codebase variable length
                planning horizons can be implemented.
        """

        self.num_buildings = N
        self.tau = tau

        # Load in pre-computed prediction model.
        # ====================================================================
        # insert your loading code here
        # ====================================================================

        # Create buffer/tracking attributes
        self.prev_observations = None
        self.buffer = {'key': []}
        # ====================================================================


        # dummy forecaster buffer - delete for your implementation
        # ====================================================================
        self.prev_vals = {'loads': None, 'pv_gens': None, 'price': None, 'carbon': None}
        # ====================================================================


    def compute_forecast(self, observations):
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

        # ====================================================================
        # insert your forecasting code here
        # ====================================================================


        # dummy forecaster for illustration - delete for your implementation
        # ====================================================================
        current_vals = {
            'loads': np.array(observations)[:,20],
            'pv_gens': np.array(observations)[:,21],
            'pricing': np.array(observations)[0,24],
            'carbon': np.array(observations)[0,19]
        }


        if self.prev_vals['carbon'] is None:
            predicted_loads = np.repeat(current_vals['loads'].reshape(self.num_buildings,1),self.tau,axis=1)
            predicted_pv_gens = np.repeat(current_vals['pv_gens'].reshape(self.num_buildings,1),self.tau,axis=1)
            predicted_pricing = np.repeat(current_vals['pricing'],self.tau)
            predicted_carbon = np.repeat(current_vals['carbon'],self.tau)

        else:
            predict_inds = [t+1 for t in range(self.tau)]

            # note, pricing & carbon predictions of all zeros can lead to issues, so clip to 0.01
            load_lines = [np.poly1d(np.polyfit([-1,0],[self.prev_vals['loads'][b],current_vals['loads'][b]],deg=1)) for b in range(self.num_buildings)]
            predicted_loads = np.array([line(predict_inds) for line in load_lines]).clip(0.01)

            pv_gen_lines = [np.poly1d(np.polyfit([-1,0],[self.prev_vals['pv_gens'][b],current_vals['pv_gens'][b]],deg=1)) for b in range(self.num_buildings)]
            predicted_pv_gens = np.array([line(predict_inds) for line in pv_gen_lines]).clip(0)

            predicted_pricing = np.poly1d(np.polyfit([-1,0],[self.prev_vals['pricing'],current_vals['pricing']],deg=1))(predict_inds).clip(0.01)

            predicted_carbon = np.poly1d(np.polyfit([-1,0],[self.prev_vals['carbon'],current_vals['carbon']],deg=1))(predict_inds).clip(0.01)


        self.prev_vals = current_vals
        # ====================================================================


        return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon
