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
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner
from utils.pat import model_finder, Data
from lightning.pytorch.callbacks.progress import TQDMProgressBar
torch.set_float32_matmul_precision('medium')


class Predictor:
    def __init__(self, mparam_dict=None, building_indices=(5, 11, 14, 16, 24, 29), L=144, T=48, expt_name='linear',
                 load=False):
        """Initialise Prediction object and perform setup.
        
        Args:
            mparam (dict): todo
            building_indices (typle of int): todo
                requiring forecasts.
            T (int): length of planning horizon (number of time instances
                into the future to forecast).
                Note: with some adjustment of the codebase variable length
                planning horizons can be implemented.
            L (int):    todo
        """
        if mparam_dict is None:
            mparam_dict = {'all': {'model_name': 'vanilla',
                                   'mparam': {'L': L,
                                              'T': T,
                                              'layers': []}
                                   }
                           }
        valid_types = ('all', 'solar', 'load', 'carbon', 'price')
        error_str = f'incorrect keys provided in mparam_dict, only the following is allowed: {valid_types}'
        assert all([key in valid_types for key in mparam_dict.keys()]), error_str

        if load:
            with open(os.path.join(expt_name, 'mparam_dict.json'), 'r') as file:
                mparam_dict = json.load(file)
                mparam = next(iter(mparam_dict.values()))['mparam']
                L = mparam['L']
                T = mparam['T']
        else:
            assert not os.path.exists(expt_name), 'expt_name already taken'
            os.makedirs(expt_name)
            with open(os.path.join(expt_name, 'mparam_dict.json'), 'w') as file:
                json.dump(mparam_dict, file)

        self.mparam_dict = mparam_dict
        self.building_indices = building_indices
        self.T = T
        self.L = L
        self.expt_name = expt_name

        self.training_order = [f'solar_{b}' for b in building_indices]
        self.training_order += [f'load_{b}' for b in building_indices]
        self.training_order += ['carbon', 'price']

        self.models = {}
        if 'all' in mparam_dict.keys():
            for key in self.training_order:
                self.models[key] = model_finder(self.mparam_dict['all']['model_name'],
                                                self.mparam_dict['all']['mparam'])
        else:
            for key in self.mparam_dict.keys():
                self.models[key] = model_finder(self.mparam_dict[key]['model_name'],
                                                self.mparam_dict[key]['mparam'])

        self.buffer = {}    # todo

    def train(self, patience=25, max_epoch=200):
        for key in self.training_order:
            self.train_individual(key, patience, max_epoch)

    def train_individual(self, key, patience=25, max_epoch=200):
        # todo
        # key is of the form 'solar_5'

        if '_' in key:  # deal with solar and load
            dataset_type, building_index = key.split('_')
        else:   # deal with carbon and price
            building_index = self.building_indices[0]
            dataset_type = key

        # datasets
        train_dataset = Data(building_index, self.L, self.T, dataset_type, 'train')
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataset = Data(building_index, self.L, self.T, dataset_type, 'validate')
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # early stop
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            min_delta=1e-6,
                                            patience=patience,
                                            verbose=False,
                                            mode="min")

        # status report
        status = f'{key} ({self.training_order.index(key) + 1} / {len(self.training_order)})'

        class CustomProgressBar(TQDMProgressBar):
            def get_metrics(self, *args, **kwargs):
                items = super().get_metrics(args[0], args[1])
                items.pop("v_num", None)
                items['current dataset'] = status
                return items

        # train the model
        model = self.models[key]
        logger = TensorBoardLogger(f'{self.expt_name}/', name=key)
        trainer = Trainer(max_epochs=max_epoch,
                          logger=logger,
                          accelerator="cuda",
                          devices=find_usable_cuda_devices(1),
                          log_every_n_steps=10,
                          callbacks=[early_stop_callback, CustomProgressBar()])
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloader, val_dataloader, min_lr=1e-6, max_lr=1e-1)
        lr = lr_finder.suggestion()
        model.learning_rate = lr
        trainer.fit(model, train_dataloader, val_dataloader)

    def test_individual(self, building_index=5, dataset_type='solar'):
        key = dataset_type
        if dataset_type not in ('price', 'carbon'):
            key += '_' + str(building_index)
        expt_dir = os.path.join(self.expt_name, key, f'version_0', 'checkpoints')
        checkpoint_name = os.listdir(expt_dir)[0]
        load_path = os.path.join(expt_dir, checkpoint_name)

        test_dataset = Data(building_index=building_index, L=self.L, T=self.T,
                            version='test', dataset_type=dataset_type)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = self.models[key]
        if 'all' in self.mparam_dict.keys():
            mparam = self.mparam_dict['all']['mparam']
        else:
            mparam = self.mparam_dict[key]['mparam']
        model = model.load_from_checkpoint(load_path, **mparam)
        model.eval()

        pred_list = []
        x_list = []
        loss_list = []
        for x, y in test_dataloader:
            x_list.append(x[:, -1])
            y_hat = model(x)
            pred_list.append(y_hat.detach().numpy())
            error = (y[:, 0] - y_hat[:, 0]) ** 2
            loss_list.append(error.detach().numpy())

        x = np.concatenate(x_list)
        pred = np.concatenate(pred_list)
        loss = np.concatenate(loss_list)

        mse = np.mean(loss)
        print(f'mse = {mse}')
        return x, pred

    def compute_forecast(self, observations):   # todo: inference, padding with validation set
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

