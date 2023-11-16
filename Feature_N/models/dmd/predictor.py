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
import os
import csv
import json
from collections import deque

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pydmd import DMD, DMDc, HODMD, ModesTuner
from pydmd.plotter import plot_eigs
from models.dmd.utils import Data

from citylearn.citylearn import CityLearnEnv

class Predictor:

    def __init__(self, tau: int = 48, L: int = 720, building_indices=(5, 11, 14, 16, 24, 29),
                    dataset_dir=os.path.join('data', 'analysis')):
        """Initialise Prediction object and perform setup.

        Args:
            T (int): Length of planning horizon (number of time instances into the future to forecast).
            L (int): The length of the input sequence.
            building_indices (tuple of int): Indices of buildings to get data for.
        """

        self.tau = tau
        self.L = L
        self.building_indices = building_indices
        self.training_order = [f'load_{b}' for b in building_indices]
        self.training_order += ['solar','carbon','price']

        # Load in pre-computed prediction models for different type of DMD.
        # ====================================================================
        self.dmd = DMD(svd_rank=1)

        self.dmdc = DMDc(svd_rank=-1, opt=True, svd_rank_omega=1)

        self.hodmd_250 = HODMD(svd_rank=0.99,  exact=True, opt=True, d=250, forward_backward=True,
                      sorted_eigs='real')
        self.hodmd_280 = HODMD(svd_rank=0.99,  exact=True, opt=True, d=280, forward_backward=True,
                      sorted_eigs='real')
        self.hodmd_300 = HODMD(svd_rank=0.99,  exact=True, opt=True, d=300, forward_backward=True,
                      sorted_eigs='real')
        # ====================================================================

        # Create buffer/tracking attributes
        self.buffer = {}
        self.control_buffer= {}
        self.controlInputs = ['diff_solar', 'dir_solar']
        self.dif_irads = None
        self.dir_irads = None
        self.minmaxscaler = None

        # Load validation data for each building being predicted
        building_dfs = {}
        for b_id in self.building_indices:
            building_dfs[b_id] = Data(b_id, self.L, self.tau,
                                      control_inputs=self.controlInputs,
                                      version='validate',
                                      dataset_dir=dataset_dir)

        # initial condition (t=0): load data from validation set into observations buffer
        for key in self.training_order:
            building_index, dataset_type = self.key2bd(key)
            building_dataset = building_dfs[int(building_index)]
            x = building_dataset.data[building_dataset.columns.index(dataset_type)][-self.L:]
            self.buffer[key] = deque(x, maxlen=len(x))

        for control_input in self.controlInputs: # load data from validation set into control buffer
            building_index = self.building_indices[0]
            building_dataset = building_dfs[building_index]
            x = building_dataset.data[building_dataset.columns.index(control_input)][-self.L:]
            self.control_buffer[control_input] = deque(x, maxlen=len(x))


    def key2bd(self, key):
        """Extracts the building index and dataset type from a given key.

        Args:
            key (str): A string containing the dataset type and building index, separated by an underscore.
                Example: 'load_5', 'load_11', 'solar', 'carbon', 'price'.

        Returns:
            Tuple[int, str]: A tuple containing the building index (int) and dataset type (str).
                Example: ('load', 5), ('load', 11), ('carbon', 5), ('solar', 5)

        Notes:
            'carbon', 'price' and 'solar is shared between the buildings so will return the same building index.
        """
        if '_' in key:  # load
            dataset_type, building_index = key.split('_')
        else:  # solar, carbon, price
            building_index = self.building_indices[0]
            dataset_type = key
        return building_index, dataset_type

    def bd2key(self, building_index, dataset_type):
        """Constructs a key string from a given building index and dataset type.

        Args:
            building_index (int): An integer representing the index of the building.
            dataset_type (str): A string representing the type of dataset. It can be one of the following:
                'solar', 'load', 'price', or 'carbon'.

        Returns:
            str: A string representing the key, in the format "<dataset_type>_<building_index>" ('load')
                or "<dataset_type>" ('solar', 'carbon', 'price).
        """
        key = dataset_type
        if dataset_type == 'load':
            key += '_' + str(building_index)
        return key


    def initialise_forecasting(self, env: CityLearnEnv, use_forecast_buffer=True):

        self.use_forecast_buffer = use_forecast_buffer
        self.forecasts_buffer = {key:[] for key in self.training_order}

        #self.env = env
        self.simulation_duration = env.time_steps
        self.b0_pv_cap = env.buildings[0].pv.nominal_power

        self.dif_irads = env.buildings[0].weather.diffuse_solar_irradiance
        self.dir_irads = env.buildings[0].weather.direct_solar_irradiance


    def compute_forecast(self, observations, t: int):
        """Compute forecasts given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation.
                The observation is a list of observations for each building (sub-list),
                where the sub-lists contain values as specified in the ReadMe.md

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

        '''
        observations will be used as training data for the DMD
        observations are updated on each timestep during control loop in assess forecast 
        '''
        current_obs = {
            'load': np.array(observations)[:, 20],
            'solar': np.array(observations)[0, 21].reshape(1)*1000/self.b0_pv_cap,
            'carbon': np.array(observations)[0, 19].reshape(1),
            'price': np.array(observations)[0, 24].reshape(1)
        }
        current_obs_inp = {
            'diff_solar': np.array(observations)[0, 11].reshape(1),
            'dir_solar': np.array(observations)[0, 15].reshape(1)
        }

        # update buffers with new observations
        for key in self.training_order:
            building_index, dataset_type = self.key2bd(key)
            self.buffer[key].append(current_obs[dataset_type][self.building_indices.index(int(building_index))])
        for key in self.controlInputs:
            self.control_buffer[key].append(current_obs_inp[key][0])

        # opt out of prediction if buffer not yet full
        if any([len(self.buffer[key]) < self.L for key in self.training_order]):
            return None


        # Perform forecasting
        # ====================================================================
        out = {'solar': [], 'load': [], 'carbon': [], 'price': []}

        # use forecast buffer if enough forecasted observations remain
        if all([len(self.forecasts_buffer[key]) >= self.tau+1 for key in self.training_order]) and self.use_forecast_buffer:
            for key in self.training_order:
                building_index, dataset_type = self.key2bd(key)
                forecast = self.forecasts_buffer[key]
                out[dataset_type].append(forecast[:self.tau])
                self.forecasts_buffer[key] = self.forecasts_buffer[key][1:] # remove first element from buffer

        else: # make a forecast & refill forecast buffer

            self.forecasts_buffer = {}

            for key in self.training_order:
                building_index, dataset_type = self.key2bd(key)
                snapshots = np.array([list(self.buffer[key])[-self.L:]])  # get last L observations from buffer

                if dataset_type == 'solar':
                    '''
                    Fit DMDc to snapshots and control inputs
                    '''

                    controlInputs = np.array([list(self.control_buffer[key])[-self.L:] for key in self.controlInputs])

                    # TODO: think about normalisation
                    # normalise inputs
                    # scaler = MinMaxScaler(feature_range=(0,1))
                    # scaler = scaler.fit(values_tr)
                    # normalized_snp = scaler.fit_transform(values_tr)

                    #print(t)
                    #print(snapshots, np.max(snapshots), controlInputs, np.max(controlInputs), snapshots.shape, controlInputs.shape, snapshots[:,-96:], controlInputs[:,-96:])

                    dmd_container = self.dmdc
                    dmd_container.fit(snapshots, controlInputs[:,:-1]) # np.vstack([snapshots,controlInputs])

                    '''
                    Optional model improvement via stabilising of eigenvalues
                    '''
                    mtuner = ModesTuner(dmd_container)
                    # mtuner.select('integral_contribution', n=30)
                    mtuner.select('stable_modes', max_distance_from_unity=1.e-2)
                    # mtuner.stabilize(inner_radius=-1.e-2, outer_radius=1.e-2)
                    tunedDMD = mtuner._dmds[0]
                    dmd_container = tunedDMD

                    # NOTE: perfect forcing forecasts is cheating somewhat
                    future_df = pd.DataFrame({ # get future control inputs - padding to length L with zeros (avoid termination error)
                        'Diffuse Solar Radiation [W/m2]': np.pad(self.dif_irads[t:t+self.L], (0,self.L-len(self.dif_irads[t:t+self.L])), mode='constant'),
                        'Direct Solar Radiation [W/m2]': np.pad(self.dir_irads[t:t+self.L], (0,self.L-len(self.dir_irads[t:t+self.L])), mode='constant')
                    })
                    forecast_controlInput = future_df[['Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']].to_numpy()
                    forecast_controlInput = forecast_controlInput[:-1].T

                    '''
                    Forecast for 2*tau (double tau so that we can
                    add one on each timestep to maintain a forecast of tau hours)
                    '''

                    reconstruction = dmd_container.reconstructed_data().real # reconstruct data over training window
                    full_forecast = dmd_container.reconstructed_data(forecast_controlInput).real # forecast for L periods using future control inputs
                    # denormalised_ forecast = scaler.inverse_transform(forecast_norm)

                    forecast = full_forecast[0,:2*self.tau]

                else:
                    '''
                    Option to initialise hodmd here (but costly)
                    '''
                    # self.dmdtype='hodmd'
                    # self.dmd_container = HODMD(svd_rank=0.99, svd_rank_extra=0.9, exact=True, opt=True, d=250,
                    #               forward_backward=True,
                    #               sorted_eigs='real')
                    # self.dmd_container.fit(snapshots)

                    def setcontainer ():
                        if key == 'load_5': return self.hodmd_250
                        elif key == 'load_24': return self.hodmd_280
                        else: return self.hodmd_300

                    dmd_container = setcontainer()
                    dmd_container.fit(snapshots)
                    dmd_container.dmd_time['tend'] = (2*self.tau + self.L) - 1

                    '''
                    Optional model improvement via stabilising of eigenvalues
                    '''
                    mtuner = ModesTuner(dmd_container)
                    # mtuner.select('integral_contribution', n=30)
                    mtuner.select('stable_modes', max_distance_from_unity=1.e-1)
                    # mtuner.stabilize(inner_radius=0.5, outer_radius=1.5)
                    tunedDMD = mtuner._dmds[0]
                    dmd_container = tunedDMD

                    '''
                    Forecast for 2*tau (double tau so that we can add one on each
                    timestep to maintain a forecast of tau hours)
                    '''

                    forecast = dmd_container.reconstructed_data.real[0][self.L:self.L+2*self.tau]

                # save forecast to output
                out[dataset_type].append(forecast[:self.tau])

                # update buffer with new forecast
                self.forecasts_buffer[key] = forecast[1:] # NOTE: just used first entry from buffer so pre-pop

        # ====================================================================
        return np.array(out['load']), np.array(out['solar']).reshape(-1), np.array(out['carbon']).reshape(-1), np.array(out['price']).reshape(-1)