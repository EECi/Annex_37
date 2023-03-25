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
    def __init__(self, mparam_dict=None, building_indices=(5, 11, 14, 16, 24, 29), L=144, T=48,
                 expt_name='log_expt', results_file='results.csv', load=False):
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

        expt_dir = os.path.join('logs', expt_name)
        if load:
            with open(os.path.join(expt_dir, 'mparam_dict.json'), 'r') as file:
                mparam_dict = json.load(file)
                mparam = next(iter(mparam_dict.values()))['mparam']
                L = mparam['L']
                T = mparam['T']
        else:
            assert not os.path.exists(expt_dir), 'expt_name already taken'
            os.makedirs(expt_dir)
            with open(os.path.join(expt_dir, 'mparam_dict.json'), 'w') as file:
                json.dump(mparam_dict, file)

        self.mparam_dict = mparam_dict
        self.building_indices = building_indices
        self.T = T
        self.L = L
        self.expt_name = expt_name
        self.results_file = os.path.join('logs', results_file)

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
        """Train all models.

        Args:
            patience (int): Number of epochs with no improvement in validation loss before training is stopped early.
            max_epoch (int): Maximum number of epochs for which to train each model unless stopped early.
        """
        for key in self.training_order:
            self.train_individual(key=key, patience=patience, max_epoch=max_epoch)

    def train_individual(self, building_index=None, dataset_type=None, patience=25, max_epoch=200, key=None):
        """Train an individual model.

        Args:
            building_index (int): Index of the building for which to generate forecasts.
            dataset_type (str): Type of dataset to use for prediction. Must be one of
                ('solar', 'load', 'carbon', 'price').
            key (str): Represents the dataset type and building index (e.g. 'solar_5', 'load_5, 'price', 'carbon').
            patience (int): Number of epochs with no improvement in validation loss before training is stopped early.
            max_epoch (int): Maximum number of epochs for which to train the model.

        Note:
            price and carbon is shared between the buildings so does not have an integer to specify the building index.

            The information required to load the corresponding dataset is contained in either the 'key' (alone) or
            both the 'building_index' and 'dataset_type'. Either specify both the 'building_index' and 'dataset_type'
            or just the 'key'. If the key is specified, this will overwrite the 'building_index' and 'dataset_type'
            specified.
        """

        bd = building_index is not None and dataset_type is not None
        assert bd or key is not None, 'either specify both building_index and dataset_type, or key'

        if key is None:
            key = dataset_type
            if dataset_type not in ('price', 'carbon'):
                key += '_' + str(building_index)
        else:
            if '_' in key:  # deal with solar and load
                dataset_type, building_index = key.split('_')
            else:  # deal with carbon and price
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
        logger = TensorBoardLogger(f'{os.path.join("logs", self.expt_name)}/', name=key)
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

    def test(self):
        """Test all models and save the results to a csv..

        Example:
            results_file = 'results.csv'
            expt_name = 'logs_linear_L168_T48'
            predictor = Predictor(expt_name=expt_name, load=True, results_file=results_file)
            header, results = predictor.test()

        Notes:
            For testing, load must be set to True on Predictor instantiation. This will load the saved model given by
            the expt_name directory. The results will be saved to a file specified by 'results_file'.
        """

        header = ['experiment']
        header += self.training_order
        results = [self.expt_name]

        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        for key in self.training_order:
            status = f'{key} ({self.training_order.index(key) + 1} / {len(self.training_order)})'
            print(f'testing: {status}')
            _, _, mse, = self.test_individual(key=key)
            results.append(mse)
            print(f'mse = {mse:.4g}\n')

        with open(self.results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)
        return header, results

    def test_individual(self, building_index=None, dataset_type=None, key=None):
        """Test an individual model.

        Args:
            building_index (int): Index of the building for which to generate forecasts.
            dataset_type (str): Type of dataset to use for prediction. Must be one of
            ('solar', 'load', 'carbon', 'price').
            key (str): Represents the dataset type and building index (e.g. 'solar_5', 'load_5, 'price', 'carbon').

        Returns:
            x (ndarray): The ground truth time series values. At time index 't', the ground truth value x[t].
            pred (ndarray): The predicted time series values. At time step 't', pred[t][i] gives the prediction for
            x[t + 1 + i].

        Example:
                building_index = 5
                dataset_type = 'price'
                expt_name = 'log_linear_L144_T48'
                predictor = Predictor(expt_name=expt_name, load=True)
                x, pred, mse = predictor.test_individual(building_index, dataset_type)
                print(f'mse = {mse}')

        Notes:
              For testing, load must be set to True on Predictor instantiation. This will load the saved model given by
              the expt_name directory.

              The information required to load the corresponding dataset is contained in either the 'key' (alone) or
              both the 'building_index' and 'dataset_type'. Either specify both the 'building_index' and 'dataset_type'
              or just the 'key'. If the key is specified, this will overwrite the 'building_index' and 'dataset_type'
              specified.

        """
        bd = building_index is not None and dataset_type is not None
        assert bd or key is not None, 'either specify both building_index and dataset_type, or key'

        if key is None:
            key = dataset_type
            if dataset_type not in ('price', 'carbon'):
                key += '_' + str(building_index)
        else:
            if '_' in key:  # deal with solar and load
                dataset_type, building_index = key.split('_')
            else:  # deal with carbon and price
                building_index = self.building_indices[0]
                dataset_type = key

        expt_dir = os.path.join('logs', self.expt_name, key, f'version_0', 'checkpoints')
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
        mse = np.mean(np.concatenate(loss_list))
        return x, pred, mse

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
            predicted_pricing = np.repeat(current_vals['pricing'], self.tau)
            predicted_carbon = np.repeat(current_vals['carbon'], self.tau)

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

