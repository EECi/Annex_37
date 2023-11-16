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

from citylearn.citylearn import CityLearnEnv
from models.base_predictor_model import BasePredictorModel
import numpy as np


class ExamplePredictor(BasePredictorModel):

    def __init__(self, N: int):
        """Initialise Prediction object and perform setup.
        
        Args:
            N (int): number of buildings in model, hence number of buildings
                requiring forecasts.
        """

        self.num_buildings = N

        # Load in pre-computed prediction model.
        self.load()
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

    def load(self):
        """No loading required for trivial example model."""
        pass

    def initialise_forecasting(self, tau: int, env: CityLearnEnv):
        """Initialise attributes required to perform forecasting.

        Args:
            tau (int): length of planning horizon (number of time instances
                into the future to forecast).
                Note: with some adjustment of the codebase variable length
                planning horizons can be implemented.
            env (CityLearnEnv): CityLearnEnvironment object.
        """

        self.tau = tau
        self.b0_pv_cap = env.buildings[0].pv.nominal_power

    def compute_forecast(self, observations):
        """Compute forecasts given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation.
                The observation is a list of observations for each building (sub-list),
                where the sub-lists contain values as specified in the main/README.md

        Returns:
            predicted_loads (np.array): predicted electrical loads of buildings in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pv_gens (np.array): predicted normalised energy generations of pv
                panels in each period of the planning horizon (W/kWp) - shape (tau)
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
            'pv_gens': np.array(observations)[0,21]*1000/self.b0_pv_cap,
            'pricing': np.array(observations)[0,24],
            'carbon': np.array(observations)[0,19]
        }

        if self.prev_vals['carbon'] is None:
            predicted_loads = np.repeat(current_vals['loads'].reshape(self.num_buildings,1),self.tau,axis=1)
            predicted_pv_gens = np.repeat(current_vals['pv_gens'],self.tau)
            predicted_pricing = np.repeat(current_vals['pricing'],self.tau)
            predicted_carbon = np.repeat(current_vals['carbon'],self.tau)

        else:
            predict_inds = [t+1 for t in range(self.tau)]

            # note, pricing & carbon predictions of all zeros can lead to issues, so clip to 0.01
            load_lines = [np.poly1d(np.polyfit([-1,0],[self.prev_vals['loads'][b],current_vals['loads'][b]],deg=1)) for b in range(self.num_buildings)]
            predicted_loads = np.array([line(predict_inds) for line in load_lines]).clip(0.01)

            predicted_pricing = np.poly1d(np.polyfit([-1,0],[self.prev_vals['pv_gens'],current_vals['pv_gens']],deg=1))(predict_inds).clip(0.01)

            predicted_pricing = np.poly1d(np.polyfit([-1,0],[self.prev_vals['pricing'],current_vals['pricing']],deg=1))(predict_inds).clip(0.01)

            predicted_carbon = np.poly1d(np.polyfit([-1,0],[self.prev_vals['carbon'],current_vals['carbon']],deg=1))(predict_inds).clip(0.01)

        self.prev_vals = current_vals
        # ====================================================================

        return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon
