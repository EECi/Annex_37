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

import torch
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pydmd import DMDc
from pydmd import DMD
from pydmd import CDMD
from pydmd import HODMD
from pydmd import SpDMD
from pydmd import ModesTuner
from pydmd import utils
from pydmd.plotter import plot_eigs
import matplotlib.pyplot as plt
from string import ascii_lowercase as alc
from utils.Data import Data
from torch.utils.data import DataLoader

from citylearn.citylearn import CityLearnEnv

class Predictor:

    # def __init__(self, mparam_dict=None, building_indices=(5, 11, 14, 16, 24, 29), L=720, T=48,
    #              hodmd_d=250, expt_name='log_expt', results_file='results.csv', load=False):
    def __init__(self, tau: int, building_indices=(5, 11, 14, 16, 24, 29), L=720):
        """Initialise Prediction object and perform setup.

        Args:
            mparam_dict (dict): Dictionary containing hyperparameters for each dataset type.
                If None, defaults to a vanilla model configuration for all dataset types.
                The keys of the dictionary must be one of ('all', 'solar', 'load', 'carbon', 'price').
                The value for each key is another dictionary with the following keys:
                    - model_name (str): Name of the model class to use.
                    - mparam (dict): Dictionary of hyperparameters to pass to the model.
            building_indices (tuple of int): Indices of buildings to get data for.
            T (int): Length of planning horizon (number of time instances into the future to forecast).
            L (int): The length of the input sequence.
            expt_name (str): Name of the experiment. Used to create a directory to save experiment-related files.
            load (bool): Whether to load hyperparameters from a the directory provided by 'expt_name'. Set True for
                training and False for testing.
        """

        self.tau = tau
        self.L = L
        self.building_indices = building_indices
        self.env = None
        self.training_order = [f'load_{b}' for b in building_indices]
        self.training_order += [f'solar_{b}' for b in building_indices]
        self.training_order += ['carbon', 'price']

        # Load in pre-computed prediction models for different type of DMD.
        # ====================================================================

        self.dmd = DMD(svd_rank=1)

        self.dmdc = DMDc(svd_rank=-1, opt=True,  svd_rank_omega=1,)

        self.hodmd_250 = HODMD(svd_rank=0.99, svd_rank_extra=0.9, exact=True, opt=True, d=250, forward_backward=True,
                      sorted_eigs='real')
        self.hodmd_300 = HODMD(svd_rank=0.99, svd_rank_extra=0.9, exact=True, opt=True, d=300, forward_backward=True,
                      sorted_eigs='real')
        self.hodmd_280 = HODMD(svd_rank=0.99, svd_rank_extra=0.9, exact=True, opt=True, d=280, forward_backward=True,
                      sorted_eigs='real')
        # ====================================================================

        # Create buffer/tracking attributes
        # self.prev_observations = None
        self.forecasts_buffer = None
        self.buffer = {}
        self.control_buffer= {}
        self.controlInputs = ['diff_solar', 'dir_solar']
        self.dif_irads = None
        self.dir_irads = None
        self.minmaxscaler = None

        print ('training order \n ', self.training_order)

        # initial condition (t=0): load data from validation set into buffer
        for key in self.training_order:
            # populate buffer using validation set
            building_index, dataset_type = self.key2bd(key)
            tr_dataset = Data(building_index, self.L, self.tau, dataset_type, 'validate', control_inputs=self.controlInputs)

            # x = tr_dataset.x[-1]
            # inputs = tr_dataset.y[-1]
            x = tr_dataset[-1] #we only take the last row hence [-1]
            self.buffer[key] = deque(x, maxlen=len(x))

        if not self.control_buffer:
            tr_dataset = Data(building_index, self.L, self.tau, dataset_type, 'validate',
                              control_inputs=self.controlInputs)
            # _, y = tr_dataset
            y = tr_dataset.y
            # print ('this is y \n ', y.shape)
            y_ = y[-1] #only take last window of len L
            self.control_buffer[self.controlInputs[0]] = deque(y_[0], maxlen = len(y_[0]))
            self.control_buffer[self.controlInputs[1]] = deque(y_[1], maxlen = len(y_[1]))

            # print ('initialised buffer for ',key,' \n ',self.buffer[key])
            # print('initialised control for buffer_diff solar \n ', self.control_buffer['diff_solar'])
            # print('initialised control for buffer_dir solar \n ', self.control_buffer['dir_solar'])

        # tr_dataset_dir_solar = Data(building_index=5, L=self.L, T=self.tau, dataset_type='dir_solar', version='validate')
        # print ('dir solar \n',tr_dataset_dir_solar)

        # for key in self.controlInputs:
        #     # populate buffer using validation set
        #     building_index, dataset_type = self.key2bd(key)
        #     tr_dataset = Data(building_index, self.L, self.tau, dataset_type, 'diff_solar')

        #print ('buffer ', self.buffer)
        # ====================================================================
        # dummy forecaster buffer - delete for your implementation
        # ====================================================================
        self.prev_vals = {'loads': None, 'pv_gens': None, 'price': None, 'carbon': None}
        # ====================================================================

    def initialise_forecasting(self, env: CityLearnEnv):
        self.simulation_duration = env.time_steps
        #TODO: insert control inptus from validate set at the front (i=0)

        self.controlInputs = ['diff_solar', 'dir_solar']
        self.dif_irads = env.buildings[1].weather.diffuse_solar_irradiance
        self.dir_irads = env.buildings[1].weather.direct_solar_irradiance
        if len(self.dif_irads)==0: print ('past observations are empty)')
        self.env = env
        self.b0_pv_cap = env.buildings[0].pv.nominal_power

        df = pd.DataFrame({'a':self.dif_irads})
        # print('initalised inputs \n', self.past_dif_irads)
        # print (df)

    def compute_forecast(self, env, observations, t:int, compute_forecast=True):
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

        # update observation
        '''
        observations will be used as training data for the DMD
        observations are updated on each timestep during control loop in assess forecast 
        '''
        current_obs = {
            'solar': np.array(observations)[:, 21],
            'load': np.array(observations)[:, 20],
            'carbon': np.array(observations)[0, 19].reshape(1),
            'price': np.array(observations)[0, 24].reshape(1)
        }

        current_obs_inp = {
            'diff_solar': np.array(observations)[0, 11].reshape(1),
            'dir_solar': np.array(observations)[0, 15].reshape(1)
        }

        out = {'solar': [], 'load': [], 'carbon': [], 'price': []}


        if (len(self.buffer['load_5']) < self.L) or ((self.simulation_duration - 1) - t < self.tau):
            # print ('do not predict for t=0')
            return None  # opt out of prediction if buffer not yet full

        if compute_forecast==True:

            controlInputs = []

            for key in self.control_buffer:
                self.control_buffer[key].append(current_obs_inp[key][0]) #append current observation to buffer
                # controlInputs.append(np.array([list(self.control_buffer[key])[-self.L:]]))

            for key in self.training_order:

                print (key)

                building_index, dataset_type = self.key2bd(key)
                self.buffer[key].append(current_obs[dataset_type][self.building_indices.index(int(building_index))]) #appends training data (current observations) to buffer
                snapshots = np.array([list(self.buffer[key])[-self.L:]])  #only need last L hours from the observation set

                if dataset_type == 'solar':
                    '''
                    fit DMDc to snapshots and control inputs       
                    '''
                    # Store snapshots as dataframe
                    snapshot_df = pd.DataFrame(snapshots.T)

                    # Process solar observations to be in W/kWp
                    snapshot_df = snapshot_df.apply(lambda x:x*1000/self.b0_pv_cap)

                    # Normalise training and control input data
                    controlInputs_df = pd.DataFrame.from_dict(self.control_buffer)[-self.L:]

                    scaler = MinMaxScaler(feature_range=(0,1))
                    data_df = pd.concat([snapshot_df, controlInputs_df], axis=1)  # concatinate tr + input columns

                    values_tr = data_df.values
                    values_tr = values_tr.reshape((len(data_df), 3))  # TODO: modularise '3'

                    # scaler = StandardScaler(with_std=False, with_mean=False, copy=True)

                    # normalized_snp = scaler.fit_transform(snapshots)
                    scaler = scaler.fit(values_tr)
                    normalized_snp = scaler.fit_transform(values_tr)

                    snapshots = normalized_snp.T

                    # snapshots = data_df.to_numpy().T

                    controlInputs = controlInputs_df[self.controlInputs].to_numpy()
                    dmd_container = self.dmdc

                    # controlInput = base_df[['Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']][:self.L].to_numpy()
                    controlInputs = np.array(controlInputs) [:-1].T

                    dmd_container.fit(snapshots, controlInputs)

                    mtuner = ModesTuner(dmd_container)
                    # mtuner.select('integral_contribution', n=30)
                    # mtuner.select('stable_modes', max_distance_from_unity=1.e-2)
                    # mtuner.stabilize(inner_radius=-1.e-2, outer_radius=1.e-2)
                    tunedDMD = mtuner._dmds[0]
                    dmd_container = tunedDMD

                    future_dif_irads = self.dif_irads[t  : self.L +t ]
                    future_dir_irads = self.dir_irads[t  : self.L +t ]

                    future_df = pd.DataFrame({
                        'Diffuse Solar Radiation [W/m2]': future_dif_irads,
                        'Direct Solar Radiation [W/m2]': future_dir_irads
                    })


                    forecast_controlInput = future_df[['Diffuse Solar Radiation [W/m2]', 'Direct Solar Radiation [W/m2]']].to_numpy()
                    forecast_controlInput = forecast_controlInput [:-1].T

                    """
                    Test: normalising control inputs independently 
                    """
                    # forecast_controlInput=np.vstack([forecast_controlInput,dummy])
                    # forecast_controlInput = scaler.transform(forecast_controlInput.T).T[-2:]


                    # print('forecasting... ' ,dataset_type)
                    '''
                    forecast for tau *2 (double tau so that we can
                    add one on each timestep to maintain a forecast of tau hours)            
                    '''
                    reconstruction_norm = dmd_container.reconstructed_data().real
                    reconstructed = reconstruction_norm[0]

                    #
                    forecast_norm = dmd_container.reconstructed_data(forecast_controlInput).real
                    forecast_lifted = pd.DataFrame(scaler.inverse_transform(forecast_norm.T))

                    forecast = forecast_lifted[0][len(snapshots)-2:len(snapshots) -2+ self.tau * 2] #-2 temporarily corrects shift in forecast vs. observer (need to find issue)
                    # forecast = forecast_norm[0][len(snapshots):len(snapshots) + self.tau * 2]

                    observed_test = np.array([self.env.buildings[0].energy_simulation.solar_generation[t:t+self.tau*2]])


                    """
                    # Plot reconstructed and forecasted signals for debugging
                    
                    # fig, ax = plt.subplots()
                    # plt.ion()
                    # ax.plot(snapshots[0].T, label= 'observed tr '+ str(dataset_type))
                    # ax.plot(reconstructed, label='reconstructed '+str(dataset_type))
                    # ax.plot(np.arange(len(reconstructed), len(reconstructed)+len(forecast)), forecast, label='forecast '+str(dataset_type))
                    #
                    # ax.plot(np.arange(len(reconstructed), len(reconstructed)+ len(forecast)), observed_test.reshape(96,),
                    #         label='observed test ' + str(dataset_type))
                    # ax.legend()
                    # plt.pause(1)
                    # # input("Press Enter to continue...")
                    # plt.clf()
                    # plt.show()
                    """

                else:
                    # print ('fitting to snapshot data using HODMD')

                    '''
                    option to initialise hodmd here (but costly)         
                    '''
                    # self.dmdtype='hodmd'
                    # self.dmd_container = HODMD(svd_rank=0.99, svd_rank_extra=0.9, exact=True, opt=True, d=250,
                    #               forward_backward=True,
                    #               sorted_eigs='real')
                    # self.dmd_container.fit(snapshots)

                    def setcontainer ():
                        # print ('key container  ', key )
                        if key == 'load_5': return self.hodmd_250
                        if key == 'load_24': return self.hodmd_280
                        else: return self.hodmd_300

                    dmd_container = setcontainer()
                    dmd_container.fit(snapshots)
                    dmd_container.dmd_time['tend'] = ((self.tau + self.L)) -1

                    mtuner = ModesTuner(dmd_container)
                    # mtuner.select('integral_contribution', n=30)
                    mtuner.select('stable_modes', max_distance_from_unity=1.e-1)
                    # mtuner.stabilize(inner_radius=0.5, outer_radius=1.5)
                    tunedDMD = mtuner._dmds[0]
                    dmd_container = tunedDMD

                    # print('forecasting... ',dataset_type)
                    '''
                    forecast for tau *2 (double tau so that we can
                    add one on each timestep to maintain a forecast of tau hours)            
                    '''

                    forecast = dmd_container.reconstructed_data.real[0][len(snapshots):len(snapshots) + self.tau * 2]

                    """
                    # Plot snapshots for debugging
                    
                    # plt.ion()
                    # ax.plot(snapshots)
                    # ax.plot(forecast, label=dataset_type)
                    # ax.legend()
                    # plt.pause(1)
                    # input("Press Enter to continue...")
                    """

                # save forecast to output
                out[dataset_type].append(forecast)
                # out[dataset_type].append(self.models[key](x).detach().numpy()) # here you use the self.dmd model to forecast

            # ====================================================================
            return [np.array(out['load']), np.array(out ['solar']), np.array(out['carbon']).reshape(-1), np.array(out['price']).reshape(-1)], reconstructed
            # return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon

        else: # if compute_forecast = False

            self.control_buffer['diff_solar'].append(current_obs_inp['diff_solar'][0])
            self.control_buffer['dir_solar'].append(current_obs_inp['dir_solar'][0])

            for key in self.training_order:
                # print ('key ', key)
                building_index, dataset_type = self.key2bd(key)
                self.buffer[key].append(current_obs[dataset_type][self.building_indices.index(int(building_index))]) #appends training data (current observations) to buffer

    def key2bd(self, key):
        """Extracts the building index and dataset type from a given key.

        Args:
            key (str): A string containing the dataset type and building index, separated by an underscore.
                Example: 'load_5', 'load_11', 'carbon', 'price', 'solar_5'.

        Returns:
            Tuple[int, str]: A tuple containing the building index (int) and dataset type (str).
                Example: ('load', 5), ('load', 11), ('carbon', 5), ('solar', 5)

        Notes:
            'carbon', 'price' and 'solar is shared between the buildings so will return the same building index.
        """
        if '_' in key:  # solar, load
            dataset_type, building_index = key.split('_')
        else:  # carbon and price
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
            str: A string representing the key, in the format "<dataset_type>_<building_index>" (load, solar)
                or "<dataset_type>" ('carbon', 'price).
        """
        key = dataset_type
        if dataset_type not in ('price', 'carbon'):
            key += '_' + str(building_index)
        return key