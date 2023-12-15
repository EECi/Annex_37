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
from models.dmd.utils import Data
# from utils.utils import Data

from citylearn.citylearn import CityLearnEnv

import xgboost 
from sklearn.multioutput import MultiOutputRegressor
from  lightgbm import LGBMRegressor

import matplotlib.pylab as plt


class LGBMOnlinePredictor_iterative:

    def __init__(self, tau: int = 48, L: int = 720, building_indices=(5, 11, 14, 16, 24, 29),
                    dataset_dir=os.path.join('data', 'example'), lags=[12,24,48]): #analysis
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
        self.lags = lags # cummulative history

        
        # Create buffer/tracking attributes
        self.buffer = {}
        self.buffer_lag = {}
        self.control_buffer= {}
        self.control_buffer_lag= {}
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
            x = building_dataset.data[building_dataset.columns.index(dataset_type)][-self.L-self.tau:]
            self.buffer[key] = deque(x, maxlen=len(x))
            if self.lags:
                x_lag = building_dataset.data[building_dataset.columns.index(dataset_type)][-self.L-self.tau-max(self.lags):]
                self.buffer_lag[key] = deque(x_lag, maxlen=len(x_lag))
            

        for control_input in self.controlInputs: # load data from validation set into control buffer
            building_index = self.building_indices[0]
            building_dataset = building_dfs[building_index]
            x = building_dataset.data[building_dataset.columns.index(control_input)][-self.L-self.tau:]
            self.control_buffer[control_input] = deque(x, maxlen=len(x))
            if self.lags:
                x_lag = building_dataset.data[building_dataset.columns.index(control_input)][-self.L-self.tau-max(self.lags):]
                self.control_buffer_lag[control_input] = deque(x_lag, maxlen=len(x_lag))

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


    def initialise_forecasting(self, env: CityLearnEnv):

        # self.use_forecast_buffer = use_forecast_buffer
        # self.forecasts_buffer = {key:[] for key in self.training_order}

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
            if self.lags:
                self.buffer_lag[key].append(current_obs[dataset_type][self.building_indices.index(int(building_index))])

        for key in self.controlInputs:
            self.control_buffer[key].append(current_obs_inp[key][0])
            if self.lags:
                self.control_buffer_lag[key].append(current_obs_inp[key][0])

        # opt out of prediction if buffer not yet full
        if any([len(self.buffer[key]) < self.L for key in self.training_order]):
            return None


        # Perform forecasting
        # ====================================================================
        out = {'solar': [], 'load': [], 'carbon': [], 'price': []}

        # # use forecast buffer if enough forecasted observations remain
        # if all([len(self.forecasts_buffer[key]) >= self.tau+1 for key in self.training_order]) and self.use_forecast_buffer:
        #     for key in self.training_order:
        #         building_index, dataset_type = self.key2bd(key)
        #         forecast = self.forecasts_buffer[key]
        #         out[dataset_type].append(forecast[:self.tau])
        #         self.forecasts_buffer[key] = self.forecasts_buffer[key][1:] # remove first element from buffer

        # else: # make a forecast & refill forecast buffer

            # self.forecasts_buffer = {}

        for key in self.training_order:
            building_index, dataset_type = self.key2bd(key)

            # Try predicting next time step only, and iteratively predict onwards
            # # Attempt 1
            # training_x = np.array([list(self.buffer[key])[-self.L-self.tau:-self.tau]])
            # training_y = np.array([list(self.buffer[key])[-self.tau:]]).T
            # for t in range(self.tau-1):
            #     training_x = np.concatenate([training_x, np.array([list(self.buffer[key])[-self.L-self.tau+t+1:-self.tau+t+1]])], axis=0)            
            # forecast_x = np.array([list(self.buffer[key])[-self.L:]])
            # Attempt 2            
            training_x = np.array([list(self.buffer[key])[-self.L-self.tau:-self.tau]]).T
            training_y = np.array([list(self.buffer[key])[-self.L:]]).T
            for t in range(self.tau-1):
                training_x = np.concatenate([training_x, np.array([list(self.buffer[key])[-self.L-self.tau+t+1:-self.tau+t+1]]).T], axis=1)            
            forecast_x = np.array([list(self.buffer[key])[-self.tau:]])
            
            # print(training_x.shape)
            # print(training_y.shape)
            # print(forecast_x.shape)
            
            # for l in self.lags:
            #     training_x = np.concatenate([training_x, np.array([list(self.buffer_lag[key])[-self.L-self.tau-l:-self.tau-l]]).T], axis=1)
            #     # training_x_lags = np.array([list(self.buffer_lag[key])[-self.L-self.tau-self.lag:-self.tau-self.lag]]).T
            # for l in self.lags:
            #     forecast_x = np.concatenate([forecast_x, np.array([list(self.buffer_lag[key])[-self.tau-l:-l]]).T], axis=1)
            #     # training_x_lags = np.array([list(self.buffer_lag[key])[-self.L-self.tau-self.lag:-self.tau-self.lag]]).T

            
            # print(training_x.shape)
            # print(training_y.shape)
            # print(forecast_x.shape)
            
            predictive_model = LGBMRegressor(verbose=-1) #xgboost.XGBRegressor(objective='reg:squarederror') #MultiOutputRegressor(xgboost.XGBRegressor(objective='reg:squarederror')) #self.xgb.copy()

        
            if dataset_type == 'solar_DONOTUSE':

                controlInputs_training = np.array([list(self.control_buffer[key])[-self.L-self.tau:-self.tau] for key in self.controlInputs]).T
                for t in range(self.tau-1):
                    controlInputs_training = np.concatenate([controlInputs_training, np.array([list(self.control_buffer[key])[-self.L-self.tau+t+1:-self.tau+t+1]]).T], axis=1)        
                # for l in self.lags:
                #     controlInputs_training = np.concatenate([controlInputs_training, np.array([list(self.control_buffer_lag[key])[-self.L-self.tau-l:-self.tau-l] for key in self.controlInputs]).T], axis=1)

                controlInputs_forecast = np.array([list(self.control_buffer[key])[-self.tau:] for key in self.controlInputs]).T
                # for l in self.lags:
                #     controlInputs_forecast = np.concatenate([controlInputs_forecast, np.array([list(self.control_buffer_lag[key])[-self.tau-l:-l] for key in self.controlInputs]).T], axis=1)
                

                # TODO: think about normalisation
                # normalise inputs
                # scaler = MinMaxScaler(feature_range=(0,1))
                # scaler = scaler.fit(values_tr)
                # normalized_snp = scaler.fit_transform(values_tr)

                #print(t)
                #print(snapshots, np.max(snapshots), controlInputs, np.max(controlInputs), snapshots.shape, controlInputs.shape, snapshots[:,-96:], controlInputs[:,-96:])

                # dmd_container = self.dmdc
                # dmd_container.fit(snapshots, controlInputs[:,:-1]) # np.vstack([snapshots,controlInputs])
                # print(training_x.shape)
                # print(controlInputs_training.shape)
                predictive_model.fit(np.concatenate([training_x,controlInputs_training],axis=1),np.ravel(training_y, order="c"))
                
                forecast = predictive_model.predict(np.concatenate([forecast_x,controlInputs_forecast],axis=1))

            else:
                

                
                # predictive_model = LGBMRegressor(verbose=-1) #xgboost.XGBRegressor(objective='reg:squarederror')  #MultiOutputRegressor(xgboost.XGBRegressor(objective='reg:squarederror')) #self.xgb.copy()
                # print(training_x.shape)
                # print(training_y.shape)
                predictive_model.fit(training_x,np.ravel(training_y, order="c")) # ravel due to format of lgbm algorithm
                
                # predict iteratively over tau
                # print(forecast_x)
                for t in range(self.tau):
                    f_next = predictive_model.predict(forecast_x)
                    forecast_x[0,:-1] = forecast_x[0,1:]
                    forecast_x[0,-1] = f_next
                forecast = np.ravel(forecast_x)
                # print(forecast_x)
            
            
            # plt.figure(figsize=(15,7))
            # plt.plot(training_x)
            # plt.plot(forecast)
            # plt.show()
        
            
            # save forecast to output
            out[dataset_type].append(forecast) #forecast[:self.tau]

            # update buffer with new forecast
            # self.forecasts_buffer[key] = forecast[1:] # NOTE: just used first entry from buffer so pre-pop
                
        # ====================================================================
        return np.array(out['load']), np.array(out['solar']).reshape(-1), np.array(out['carbon']).reshape(-1), np.array(out['price']).reshape(-1)