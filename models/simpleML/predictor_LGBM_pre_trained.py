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

Implementation of a predictor that loads pre-trained LGBM models and uses them
to forecast the next tau values for each variable. 


"""
import os
# import csv # did not end up using
# import json # did not end up using
from collections import deque

import numpy as np
# import pandas as pd # did not end up using
# from sklearn.preprocessing import StandardScaler, MinMaxScaler # did not end up using
from models.dmd.utils import Data
from citylearn.citylearn import CityLearnEnv

import pickle # to load pre-trained models
# from  lightgbm import LGBMRegressor # could use to re-train
# import matplotlib.pylab as plt # did not end up using


class LGBMOnlinePredictor_pre_trained:

    def __init__(self, tau: int = 48, L: int = 720, building_indices=(5, 11, 14, 16, 24, 29),
                    dataset_dir=os.path.join('data', 'example'), lags=[], iterative=True): 
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
        self.iterative = iterative # is prediction iterative
        
        # Store pre-trained models
        self.models = {}
        
        # Create buffer/tracking attributes
        self.buffer = {}
        self.buffer_lag = {}
        self.control_buffer= {}
        self.control_buffer_lag= {}
        
        if self.iterative:
            self.controlInputs = [] # iterative only predicts single variable so no control inputs
            model_name_type = "iterative"
        else:
            self.controlInputs = ['diff_solar', 'dir_solar']
            model_name_type = "non_iterative"
            

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
            x = building_dataset.data[building_dataset.columns.index(dataset_type)][-self.L-self.tau:] # use up to L + tau past data points, so that training set remains of length L
            self.buffer[key] = deque(x, maxlen=len(x))
            if self.lags:
                x_lag = building_dataset.data[building_dataset.columns.index(dataset_type)][-self.L-self.tau-max(self.lags):]
                self.buffer_lag[key] = deque(x_lag, maxlen=len(x_lag))
            # Load models
            if dataset_type=="load":
                model_name = "LGBM_" + model_name_type + "_t1_singleFeature_B" + str(building_index) +"_v1"
                self.models["B"+str(building_index)] = pickle.load(open(os.path.join("models","simpleML","pre_trained", model_name), "rb"))
            else:
                model_name = "LGBM_" + model_name_type + "_t1_singleFeature_" + dataset_type +"_v1"
                self.models[dataset_type] = pickle.load(open(os.path.join("models","simpleML","pre_trained", model_name), "rb"))
            

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

        self.simulation_duration = env.time_steps
        self.b0_pv_cap = env.buildings[0].pv.nominal_power

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
        observations will be used as input data for the prediction models
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

        # opt out of prediction if buffer not yet full - should not be the case
        if any([len(self.buffer[key]) < self.L for key in self.training_order]):
            return None

        # Perform forecasting
        # ====================================================================
        
        if self.iterative:
            return self.compute_forecast_iterative()
        else:
            return self.compute_forecast_non_iterative()
        
        
     
    def compute_forecast_iterative(self):
        
        # Perform forecasting
        # ====================================================================
        out = {'solar': [], 'load': [], 'carbon': [], 'price': []}
        
        for key in self.training_order:
            building_index, dataset_type = self.key2bd(key)
            
            forecast_x = np.array([list(self.buffer[key])[-self.L:]])
            
            if dataset_type=="load":
                model_name = "B"+building_index
            else:
                model_name = dataset_type

            # iteratively predict the next tau hours
            forecast_y = np.zeros((self.tau,)) # final array of predicted tau time steps
            for t in range(self.tau):
                f_next = self.models.get(model_name).predict(forecast_x)
                forecast_y[t] = f_next
                forecast_x[0,:-1] = forecast_x[0,1:]
                forecast_x[0,-1] = f_next
            forecast = forecast_y       
            
            # save forecast to output
            out[dataset_type].append(forecast)  
                
        # ====================================================================
        return np.array(out['load']), np.array(out['solar']).reshape(-1), np.array(out['carbon']).reshape(-1), np.array(out['price']).reshape(-1)
    
    def compute_forecast_non_iterative(self):
        
        # Perform forecasting
        # ====================================================================
        out = {'solar': [], 'load': [], 'carbon': [], 'price': []}
        
        for key in self.training_order:
            building_index, dataset_type = self.key2bd(key)

            if dataset_type == "solar": # we have control parameters if solar
                forecast_x = np.zeros((self.tau,self.L*(1+len(self.controlInputs))))
                for t in range(self.tau):
                    forecast_x[t,:self.L] = np.array([list(self.buffer[key])[-self.L-self.tau+t:-self.tau+t]])
                    for c_i, control_input in enumerate(self.controlInputs):
                        forecast_x[t,self.L*(c_i+1):self.L*(c_i+2)] = np.array([list(self.control_buffer[control_input])[-self.L-self.tau+t:-self.tau+t]])                
            else:
                forecast_x = np.zeros((self.tau,self.L))
                for t in range(self.tau):
                    forecast_x[t,:] = np.array([list(self.buffer[key])[-self.L-self.tau+t:-self.tau+t]])
            
            if dataset_type=="load":
                model_name = "B"+building_index
            else:
                model_name = dataset_type

            # (all at once) predict the next tau hours
            forecast = self.models.get(model_name).predict(forecast_x)      
            
            # save forecast to output
            out[dataset_type].append(forecast) 
                
        # ====================================================================
        return np.array(out['load']), np.array(out['solar']).reshape(-1), np.array(out['carbon']).reshape(-1), np.array(out['price']).reshape(-1)
  
        
        
        
        
    
    