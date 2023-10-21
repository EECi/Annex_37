"""
Implementation controlled noise predictor.

Prediction provided is the ground truth values is
Gaussian Random Walk (GRW) noise added. The level
of noise is controllable.
"""

from citylearn.citylearn import CityLearnEnv
from models.base_predictor_model import BasePredictorModel
import numpy as np


class GRWN_Predictor(BasePredictorModel):

    def __init__(self, env: CityLearnEnv, tau: int, noise_levels: dict):
        """Initialise Prediction object and perform setup.
        
        Args:
            env (CityLearnEnv): CityLearn environment to generate noisy
                predictions from.
            tau (int): length of planning horizon (number of time instances
                into the future to forecast).
                Note: with some adjustment of the codebase variable length
                planning horizons can be implemented.
            noise_levels (dict): dictionary of noise levels to use for each
                parameter. Values are the proprotional noise level, in range
                [0,1]. Keys are `load`, `solar`, `pricing`, `carbon`. For
                `load` the value is a dict where the keys are the names
                of the buildings (from env.buildings), and the values are the
                noise levels for each buildling.
        """

        self.env = env
        self.num_buildings = len(env.buildings)
        self.tau = tau
        self.noise_levels = noise_levels

        self.param_means = {
            'load': {b.name: np.mean(b.energy_simulation.non_shiftable_load) for b in self.env.buildings},
            'solar' : np.mean(self.env.buildings[0].energy_simulation.solar_generation),
            'pricing': np.mean(self.env.buildings[0].pricing.electricity_pricing),
            'carbon': np.mean(self.env.buildings[0].carbon_intensity.carbon_intensity)
        }

    def grwn(self, abs_noise_level: float):
        """Compute Gaussian Random Walk noise series for given absolute noise level,
        corresponding to the standard deviation of the Gaussian."""

        return np.cumsum(np.random.normal(0, abs_noise_level, self.tau))

    def param_grwn(self, param_type: str):
        """Sample Gaussian Random Walk noise series for given parameter type."""

        if param_type == 'load':
            return np.array([self.grwn(self.param_means['load'][b.name] * self.noise_levels['load'][b.name]) for b in self.env.buildings])
        elif param_type == 'solar':
            return self.grwn(self.param_means['solar'] * self.noise_levels['solar'])
        elif param_type == 'pricing':
            return self.grwn(self.param_means['pricing'] * self.noise_levels['pricing'])
        elif param_type == 'carbon':
            return self.grwn(self.param_means['carbon'] * self.noise_levels['carbon'])
        else:
            raise ValueError(f"param_type must be one of ['load', 'solar', 'pricing', 'carbon'], not {param_type}")

    def compute_forecast(self, observations: list, t: int):
        """Compute forecasts given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation.
                The observation is a list of observations for each building (sub-list),
                where the sub-lists contain values as specified in the main/README.md
            t (int): current time instance at which forecasts are being made.

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

        if len(self.env.buildings[0].carbon_intensity.carbon_intensity[t+1:t+self.tau+1]) < self.tau:
            return None
        else:
            predicted_loads = np.array(
                [b.energy_simulation.non_shiftable_load[t+1:t+self.tau+1]\
                    for b in self.env.buildings]) + self.param_grwn('load')
            predicted_pv_gens = np.array(
                self.env.buildings[0].energy_simulation.solar_generation[t+1:t+self.tau+1])\
                    + self.param_grwn('solar')
            predicted_pricing = np.array(
                self.env.buildings[0].pricing.electricity_pricing[t+1:t+self.tau+1])\
                    + self.param_grwn('pricing')
            predicted_carbon = np.array(
                self.env.buildings[0].carbon_intensity.carbon_intensity[t+1:t+self.tau+1])\
                    + self.param_grwn('carbon')

        # ====================================================================

        return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon
