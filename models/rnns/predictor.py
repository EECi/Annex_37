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
import pickle
import os
import csv
import json
import torch
import numpy as np
from collections import deque

from models.rnns.utils import model_finder, Data
from models.base_predictor_model import BasePredictorModel

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from citylearn.citylearn import CityLearnEnv

torch.set_float32_matmul_precision('medium')


class Predictor(BasePredictorModel):
    def __init__(self, mparam_dict=None, building_indices=(5, 11, 14, 16, 24, 29), L=168, T=48,
                 expt_name='linear_L168_T48', results_file='results.csv', load=True):
        if mparam_dict is \
                None:
            mparam_dict = {'all': {'model_name': 'vanilla',
                                   'mparam': {'L': L,
                                              'T': T,
                                              'layers': []}
                                   }
                           }
        valid_types = ('all', 'solar', 'load', 'carbon', 'price')
        error_str = f'incorrect keys provided in mparam_dict, only the following is allowed: {valid_types}'
        assert all([key in valid_types for key in mparam_dict.keys()]), error_str
        self.mparam_dict = mparam_dict
        self.building_indices = building_indices
        self.T = T
        self.L = L
        self.expt_name = expt_name
        self.results_file = os.path.join('models', 'rnns', 'resources', results_file)
        expt_dir = os.path.join('models', 'rnns', 'resources', self.expt_name)

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



        self.training_order = [f'load_{b}' for b in building_indices]
        self.training_order += [f'solar_{b}' for b in building_indices]
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

        if load:
            self.buffer = {}
            for key in self.training_order:
                # fill up buffer using validation set
                building_index, dataset_type = self.key2bd(key)
                val_dataset = Data(building_index, self.L, self.T, dataset_type, 'validate', self.expt_name, key)
                x, _ = val_dataset[-1]
                # self.buffer[key] = deque(x, maxlen=len(x))
                # deque_rows = [deque(row, maxlen=x.shape[1]) for row in x]
                self.buffer[key] = [deque(row, maxlen=x.shape[1]) for row in x]
                if 'all' in mparam_dict.keys():
                    mparam = self.mparam_dict['all']['mparam']
                else:
                    mparam = self.mparam_dict[dataset_type]['mparam']

                dir = os.path.join(expt_dir, key, 'version_0', 'checkpoints')
                checkpoint_name = os.listdir(dir)[0]
                load_path = os.path.join(dir, checkpoint_name)
                # Set the device to 'cpu' if CUDA is not available
                map_location = torch.device('cpu') if not torch.cuda.is_available() else None
                self.models[key] = self.models[key].load_from_checkpoint(load_path, map_location=map_location, **mparam)
                self.models[key].eval()

    def train(self, patience=25, max_epoch=200):
        """Train all models.

        Args:
            patience (int): Number of epochs with no improvement in validation loss before training is stopped early.
            max_epoch (int): Maximum number of epochs for which to train each model unless stopped early.
        """
        for key in self.training_order:
            self.train_individual(key=key, patience=patience, max_epoch=max_epoch)

    def train_individual(self, building_index=None, dataset_type=None, patience=25, max_epoch=200, key=None):
        bd = building_index is not None and dataset_type is not None
        assert bd or key is not None, 'either specify both building_index and dataset_type, or key'

        if key is None:
            key = self.bd2key(building_index, dataset_type)
        else:
            building_index, dataset_type = self.key2bd(key)

        # datasets
        train_dataset = Data(building_index, self.L, self.T, dataset_type, 'train', self.expt_name, key)
        train_dataloader = DataLoader(train_dataset, batch_size=32,
                                      shuffle=True)  # , num_workers=8 conflict with os.path.exists(expt_dir), 'expt_name already taken'
        val_dataset = Data(building_index, self.L, self.T, dataset_type, 'validate', self.expt_name, key)
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
        logger = TensorBoardLogger(f'{os.path.join("models", "rnns", "resources", self.expt_name)}', name=key)
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
            expt_name = 'linear_L168_T48'
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
            _, _, _, _, _, mse = self.test_individual(key=key)
            results.append(mse)
            print(f'mse = {mse:.4g}\n')

        with open(self.results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results)
        return header, results

    # predictor.test_individual(building_index, dataset_type)

    def test_individual(self, building_index=None, dataset_type=None, key=None):
        """Test an individual model.

        Args:
            building_index (int): Index of the building for which to generate forecasts.
            dataset_type (str): Type of dataset to use for prediction. Must be one of
            ('solar', 'load', 'carbon', 'price').
            key (str): Represents the dataset type and building index (e.g. 'solar_5', 'load_5, 'price', 'carbon').

        Returns:
            x (ndarray): The ground truth time series values. At time index 't', the ground truth value x[t].
            pred (ndarray): The predicted time series values.
                For example pred[t] gives the predictions for time steps [t+1, ..., t+T].
            gt (ndarray): The ground truth time series values.
                For example gt[t] = x[t-L, ..., t+T], where x is the ground truth.
            gt_t (ndarray): The time indices corresponding to each value in gt.
                For example gt_t[t] = [t-L, ..., t+T].
            pred_t (ndarray): The time indices corresponding to each value of pred.
                For example, pred_t[t] = [t+1, ..., t+T].
            mse (float): the mean squared error between the predictions and the groundtruth.

        Example:
            building_index = 5
            dataset_type = 'price'
            expt_name = 'linear_L168_T48_test2'   expt_name = '6lINE_L64_T24'
            predictor = Predictor(expt_name=expt_name, load=True)
            x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
            plotter = IndividualPlotter(x, pred, dataset_type, window_size=500)
            plotter.show()


        Notes:
              For testing, load must be set to True on Predictor instantiation. This will load the saved model given by
              the expt_name directory.

              The information required to load the corresponding dataset is contained in either the 'key' (alone) or
              both the 'building_index' and 'dataset_type'. Either specify both the 'building_index' and 'dataset_type'
              or just the 'key'. If the key is specified, this will overwrite the 'building_index' and 'dataset_type'
              specified.

        """

        bd = building_index is not None and dataset_type is not None
        assert bd or key is not None, 'either specify both building_index and dataset_type, or key'  # key='load_5'

        if key is None:
            key = self.bd2key(building_index, dataset_type)
        else:
            building_index, dataset_type = self.key2bd(key)

        self.type2idx = {'load': 0, 'solar': 1, 'price': 2, 'carbon': 3}
        # print(key)
        # print(dataset_type)
        self.type_idx = self.type2idx[dataset_type]  # type_idx =type2idx[dataset_type]
        expt_name = self.expt_name

        test_dataset = Data(building_index=building_index, L=self.L, T=self.T,
                            version='test', dataset_type=dataset_type, expt_name=expt_name, key=key)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model = self.models[key]
        x_list = []
        loss_list = []
        pred_list = []
        gt_list = []
        gt_t_list = []
        pred_t_list = []
        t = 0
        dataset_params_path_y = os.path.join(
            f'{os.path.join("models", "rnns", "resources", expt_name, key, "scalar_y_" + str(key) + ".pkl")}')
        scaler_y = pickle.load(open(dataset_params_path_y, 'rb'))
        for x, y in test_dataloader:
            # x y shape (batch, sequence length, features)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            x = x.to(device)
            y_hat = model(x)
            # y_hat.shape
            if type(y_hat) is tuple:
                y_hat = y_hat[0]  # discard the uncertainty values
            y_hat_np = y_hat.detach().cpu().numpy()  # print(y_hat.shape) #(32, 48, 1) (baict_size, length of sequence ,feature)
            y_np = y.numpy()  # print(y_np.shape) #(32, 48, 1) (baict_size, length of sequence ,feature)
            y_hat_transformed = np.zeros_like(y_hat_np)
            y_transformed = np.zeros_like(y_np)
            # scaler_y_    y_hat_np[0,:, :].shape
            for i in range(y_hat_np.shape[0]):
                y_hat_transformed[i, :, :] = scaler_y.inverse_transform(y_hat_np[i, :, :])
                y_transformed[i,:, :] = scaler_y.inverse_transform(y_np[i,:, :]) #x.shape
            x_in = x[:,:, self.type_idx] #x_in.shape
            x_np_transformed = scaler_y.inverse_transform(x_in.cpu().reshape(-1, 1)) #x_in.reshape(-1, 1).shape
            x_np_transformed = x_np_transformed.reshape(x_in.cpu().shape[0], x_in.cpu().shape[1]) # x_np_transformed.shape
            x_list.append(x_np_transformed[:, -1]) #x_np_transformed[:, -1].shape y_hat_transformed.shape
            pred_list.append(y_hat_transformed) #y_hat_transformed[:, 0, 0].shape y_hat_transformed.shape
            error = (y_transformed[:, 0, 0]- y_hat_transformed[:, 0, 0]) ** 2
            # print(y_transformed[:, :, 0].squeeze().shape)
            # print(y_transformed[:, :, 0].shape)
            loss_list.append(error)
            gt = np.concatenate([x_np_transformed, y_transformed.squeeze(axis=2)], axis=1) #gt.shape
            gt_list.append(gt)
            gt_t_list.append([np.arange(t + i, t + i + gt.shape[1]) for i in range(x.shape[0])])
            pred_t_list.append(
                [np.arange(t + i + x.shape[1], t + i + x.shape[1] + y.shape[1]) for i in range(x.shape[0])])
            t += x.shape[0] #pred_t_list.shape np.concatenate(pred_list).shape

        gt = np.concatenate(gt_list, axis=0)
        gt_t = np.concatenate(gt_t_list, axis=0)
        pred_t = np.concatenate(pred_t_list, axis=0)
        x = np.concatenate(x_list)
        pred = np.concatenate(pred_list).squeeze(2)
        mse = np.mean(np.concatenate(loss_list))
        return x, pred, gt, gt_t, pred_t, mse

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

    def compute_forecast(self, observations):
        """Compute forecasts given current observation.

        Args:
            observation (List[List]): observation data for current time instance, as
                specified in CityLearn documentation.
                The observation is a list of observations for each building (sub-list),
                where the sub-lists contain values as specified in the ReadMe.md

        Returns:
            predicted_loads (np.array): predicted electrical loads of buildings in each
                period of the planning horizon (kWh) - shape (N, tau)
            predicted_pv_gens (np.array): predicted energy generations of pv panels in each
                period of the planning horizon (kWh) - shape (N, tau)
            predicted_pricing (np.array): predicted grid electricity price in each period
                of the planning horizon ($/kWh) - shape (tau)
            predicted_carbon (np.array): predicted grid electricity carbon intensity in each
                period of the planning horizon (kgCO2/kWh) - shape (tau)
        """
        index_b = 1  # self.building_indices.index(int(building_index))

        #
        # len(predictor.buffer['load_5'][14])
        #
        # current_obs['carbon'].shape
        out = {'solar': [], 'load': [], 'carbon': [], 'price': []}
        for key in self.training_order:
            building_index, dataset_type = self.key2bd(key)
            current_obs = {
                'load': np.array(observations)[:, 20][self.building_indices.index(int(building_index))].reshape(1),
                'solar': np.array(observations)[:, 21][self.building_indices.index(int(building_index))].reshape(1),
                'price': np.array(observations)[0, 24].reshape(1),
                'carbon': np.array(observations)[0, 19].reshape(1),
                'Tem': np.array(observations)[:, 3][0].reshape(1),
                'Hum': np.array(observations)[:, 7][0].reshape(1),
                'DifSolar': np.array(observations)[:, 11][0].reshape(1),
                'DirSolar': np.array(observations)[:, 15][0].reshape(1),
                'month_sin': np.sin(2 * np.pi * np.array(observations)[:, 0][0] / 12).reshape(1),
                'month_cos': np.cos(2 * np.pi * np.array(observations)[:, 0][0] / 12).reshape(1),
                'day_sin': np.sin(2 * np.pi * np.array(observations)[:, 0][1] / 31).reshape(1),
                'day_cos': np.cos(2 * np.pi * np.array(observations)[:, 0][1] / 31).reshape(1),
                'hour_sin': np.sin(2 * np.pi * np.array(observations)[:, 0][2] / 24).reshape(1),
                'hour_cos': np.cos(2 * np.pi * np.array(observations)[:, 0][2] / 24).reshape(1),
                # 'daylight': np.array(0).reshape(1)
            }
            dataset_params_path_x = os.path.join(
                f'{os.path.join("models", "rnns", "resources", self.expt_name, key, "scalar_x_" + str(key) + ".pkl")}')
            scaler_x = pickle.load(open(dataset_params_path_x, 'rb'))
            grouped_array = np.concatenate([np.array([value]) for value in current_obs.values()])
            grouped_array=scaler_x.transform(grouped_array)
            #buffer is normalised, need to normalise the oversation as well
            np.save("grouped_array.npy", grouped_array)
            # with open("my_list1.pkl", "wb") as f:
            #     pickle.dump(self.buffer[key], f)
            for i, value in enumerate(grouped_array):
                self.buffer[key][i].append(value[0])
            # print(type(self.buffer[key]))
            # print(self.buffer[key])

            x = torch.tensor(self.buffer[key], dtype=torch.float32)
            x[0]
            x_reshape = x.reshape(1,x.shape[0], x.shape[1])
            dataset_params_path_y = os.path.join(
                f'{os.path.join("models", "rnns", "resources", self.expt_name, key, "scalar_y_" + str(key) + ".pkl")}')
            scaler_y = pickle.load(open(dataset_params_path_y, 'rb'))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x_reshape = x_reshape.to(device)
            output = self.models[key].to(device)(x_reshape)
            predict_y = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else output[
                0].detach().cpu().numpy()  # disgrad the uncertainty
            # print(output[1].detach().numpy().shape)
            predict_y_transformed = np.zeros_like(predict_y)
            # print(predict_y_transformed.shape)
            for i in range(predict_y.shape[-1]): #i=0
                predict_y_transformed[:, :, i] = scaler_y.inverse_transform(predict_y[:, :, i])
            out[dataset_type].append(predict_y_transformed.squeeze(2))  #
            # with open("out.pkl", "wb") as f:
            #     pickle.dump(out, f)

        load = np.array(out['load']).squeeze(1)
        solar = np.array(out['solar']).squeeze(1)
        price = np.array(out['price']).reshape(-1)
        carbon = np.array(out['carbon']).reshape(-1)
        return load, solar, price, carbon
