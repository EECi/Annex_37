import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import Dataset
import lightning.pytorch as pl
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pytorch_forecasting import TimeSeriesDataSet
from typing import Any, List, Dict, Union
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
import pickle
# filer warnings: mostly object reload & numpy deprecation warnings
import warnings

warnings.filterwarnings(action='ignore', module=r'pytorch_forecasting')


def model_finder(model_name, mparam):
    if model_name == 'linear':
        return LinearModel(**mparam)
    elif model_name == 'lstm':
        return LSTMmodel(**mparam)
    elif model_name == 'tsfm':
        return TransformerModel(**mparam)
    elif model_name == 'gru':
        return GRUModel(**mparam)
    elif model_name == 'DeepAR':
        return DeepAR(**mparam)

    else:
        return None


class TransformerModel(pl.LightningModule):
    def __init__(self, L, T, input_size=14, num_layers=2, hidden_size=100, learning_rate=1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = L
        self.forecast_horizon = T
        self.encoder = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=10, dim_feedforward=hidden_size * 4),
            num_layers=num_layers
        )
        self.relu = nn.ReLU()  # ReLU activation
        self.fc = nn.Linear(hidden_size, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.encoder(x)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization and transpose back
        out = self.transformer(x)
        out = out[:, -self.forecast_horizon:, :]  # Only take the last forecast_horizon outputs of the sequence
        out = self.relu(out)  # apply ReLU activation
        out = self.fc(out)
        return out.view(x.size(0), self.forecast_horizon, -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LSTMmodel(pl.LightningModule):
    def __init__(self, L, T, input_size=14, num_layers=2, hidden_size=100, learning_rate=1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = L
        self.forecast_horizon = T
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc = nn.Linear(hidden_size, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization and transpose back
        out = out[:, -self.forecast_horizon:, :]  # Only take the last forecast_horizon outputs of the sequence
        out = self.relu(out)  # apply ReLU activation
        out = self.fc(out)
        return out.view(x.size(0), self.forecast_horizon, -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class GRUModel(pl.LightningModule):
    def __init__(self, L, T, input_size=14, num_layers=2, hidden_size=100, learning_rate=1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = L
        self.forecast_horizon = T
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc = nn.Linear(hidden_size, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.batch_norm(out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization and transpose back
        out = out[:, -self.forecast_horizon:, :]  # Only take the last forecast_horizon outputs of the sequence
        out = self.relu(out)  # apply ReLU activation
        out = self.fc(out)
        return out.view(x.size(0), self.forecast_horizon, -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class DeepAR(pl.LightningModule):
    def __init__(self, L, T, input_size=14, num_layers=2, hidden_size=100, learning_rate=1e-3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = L
        self.forecast_horizon = T
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Add batch normalization layer
        self.fc_mu = nn.Linear(hidden_size, 1)
        self.fc_sigma = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.learning_rate = learning_rate

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out.transpose(1, 2)).transpose(1, 2)  # Apply batch normalization and transpose back
        out = out[:, -self.forecast_horizon:, :]  # Only take the last forecast_horizon outputs of the sequence
        out = self.relu(out)  # apply ReLU activation
        mu = self.fc_mu(out)
        sigma = self.softplus(self.fc_sigma(out))  # ensure sigma is positive
        return mu.view(x.size(0), self.forecast_horizon, -1), sigma.view(x.size(0), self.forecast_horizon, -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        dist = torch.distributions.Normal(mu, sigma)
        loss = -dist.log_prob(y).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        mu, sigma = self(x)
        dist = torch.distributions.Normal(mu, sigma)
        loss = -dist.log_prob(y).mean()
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LinearModel(pl.LightningModule):
    def __init__(self, L, T, input_size=14, output_size=1, learning_rate=1e-3):
        super().__init__()
        self.seq_len = L
        self.forecast_horizon = T
        self.linear = nn.Linear(input_size * L, output_size * T)
        self.batch_norm = nn.BatchNorm1d(input_size * L)  # Add a batch normalization layer
        self.learning_rate = learning_rate

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten the input
        x = self.batch_norm(x)  # Apply batch normalization
        out = self.linear(x)
        return out.view(batch_size, self.forecast_horizon, -1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        print(y_hat.shape)
        print(y.shape)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Data(Dataset):

    def __init__(self, building_index=5, L=48, T=24, dataset_type='load', version='train', expt_name='linear_L168_T48',
                 key='load_5'):
        super().__init__()
        self.dataset_type = dataset_type  # x,y=
        self.type2idx = {'load': 0, 'solar': 1, 'price': 2, 'carbon': 3}
        self.type_idx = self.type2idx[dataset_type]  # type_idx =type2idx[dataset_type]
        self.expt_name = expt_name
        self.key = key
        dataset_dir = os.path.join('data', 'example')
        building = pd.read_csv(os.path.join(dataset_dir, version, f'UCam_Building_{building_index}.csv'))
        load = building['Equipment Electric Power [kWh]']
        solar = building['Solar Generation [W/kW]']
        price = pd.read_csv(os.path.join(dataset_dir, version, 'pricing.csv'))['Electricity Pricing [£/kWh]']
        carbon = pd.read_csv(os.path.join(dataset_dir, version, 'carbon_intensity.csv'))
        weather = pd.read_csv(os.path.join(dataset_dir, version, 'weather.csv')).iloc[:, :4]
        #  data_categoricals = building[['Month', 'Hour', 'Day Type', 'Daylight Savings Status']].copy()
        data_categoricals = building[['Month', 'Hour', 'Day Type']].copy()
        data_numerical = pd.concat([load, solar, price, carbon, weather], axis=1)
        data_numerical.columns = ['load', 'solar', 'price', 'carbon', 'Tem', 'Hum', 'DifSolar', 'DirSolar']
        # Cyclical encoding for month, day, and hour weather.columns
        data_categoricals.loc[:, 'month_sin'] = np.sin(2 * np.pi * data_categoricals.loc[:, 'Month'] / 12)
        data_categoricals.loc[:, 'month_cos'] = np.cos(2 * np.pi * data_categoricals.loc[:, 'Month'] / 12)
        data_categoricals.loc[:, 'day_sin'] = np.sin(2 * np.pi * data_categoricals.loc[:, 'Day Type'] / 31)
        data_categoricals.loc[:, 'day_cos'] = np.cos(2 * np.pi * data_categoricals.loc[:, 'Day Type'] / 31)
        data_categoricals.loc[:, 'hour_sin'] = np.sin(2 * np.pi * data_categoricals.loc[:, 'Hour'] / 24)
        data_categoricals.loc[:, 'hour_cos'] = np.cos(2 * np.pi * data_categoricals.loc[:, 'Hour'] / 24)
        # data_categoricals.loc[:, 'daylight'] = data_categoricals.loc[:, 'Daylight Savings Status'].astype(str)
        data_categoricals = data_categoricals.drop(['Month', 'Hour', 'Day Type'], axis=1)
        #  data_categoricals = data_categoricals.drop(['Month', 'Hour', 'Day Type', 'Daylight Savings Status'], axis=1)
        data = pd.concat([data_numerical, data_categoricals],
                         axis=1)  # data.shape  data[dataset_type].values.reshape(-1, 1).shape
        data.index.name = 'time_index'
        scaler_x = StandardScaler()  # MinMaxScaler(feature_range=(0, 1))
        scaled_data_x = scaler_x.fit_transform(data)
        # print(scaled_data_x.shape) #(length, feature)
        scaler_y = StandardScaler()  # MinMaxScaler(feature_range=(0, 1))
        scaled_data_y = scaler_y.fit_transform(data[self.dataset_type].values.reshape(-1, 1))
        # print(scaled_data_y.shape) #(length, feature)
        dataset_params_dir = os.path.join("models", "rnns", "resources", self.expt_name, self.key)
        os.makedirs(dataset_params_dir, exist_ok=True)
        dataset_params_path_x = os.path.join(
            f'{os.path.join("models", "rnns", "resources", self.expt_name, self.key, "scalar_x_" + str(self.key) + ".pkl")}')
        with open(dataset_params_path_x, 'wb') as pkl_file:
            pickle.dump(scaler_x, pkl_file)
        dataset_params_path_y = os.path.join(
            f'{os.path.join("models", "rnns", "resources", self.expt_name, self.key, "scalar_y_" + str(self.key) + ".pkl")}')
        with open(dataset_params_path_y, 'wb') as pkl_file:
            pickle.dump(scaler_y, pkl_file)

        self.x = []
        self.y = []
        for i in range(len(data)):
            if i == len(data) - L - T + 1:
                break
            else:
                self.x.append(scaled_data_x[i:i + L])
                self.y.append(scaled_data_y[i + L:i + L + T])  # Select the first time series (i.e., 'load')
        self.x = np.array(self.x, dtype=np.float32)  # shape(batch size, sequence length, feature number)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print(self.x[idx].shape)
        # print(self.y[idx].shape)
        return self.x[idx], self.y[idx]  # shape (sequence length, feature number)


class IndividualPlotter:
    """A class that creates an interactive plot to visualize the predicted and ground truth values of a time series.
    Args:
        x (ndarray): The ground truth time series values. At time index 't', the ground truth value x[t]. Must be 1-dimensional.
        pred (ndarray): The predicted time series values. At time step 't', pred[t][i] gives the prediction for x[t + 1 + i].  Must
            be 2-dimensional.
        dataset_type (str): The name of the dataset type - used for y-axis label.
        window_size (int): The number of time steps to show in the plot at a time. Default is 500.
    Example:
        building_index = 5
        dataset_type = 'price'
        expt_name = 'linear_L168_T48_test2'
        predictor = Predictor(expt_name=expt_name, load=True)
        x, pred, _, _, _, mse = predictor.test_individual(building_index, dataset_type)
        plotter = IndividualPlotter(x, pred, dataset_type, window_size=500)
        plotter.show()
    Notes:
        The plot is interactive and allows the user to control the current window in the time series using a slider
        widget.
    """

    def __init__(self, x, pred, dataset_type, window_size=500):
        self.time_idx = np.arange(0, len(x))
        self.x = x
        self.pred = pred
        self.dataset_type = dataset_type

        self.window_size = window_size
        self.start_index = 0
        self.end_index = self.start_index + self.window_size

        self.fig, self.ax = plt.subplots(figsize=[15, 5])
        plt.subplots_adjust(left=0.1, bottom=0.25)
        self.alpha_list = np.linspace(1, 0, pred.shape[1])
        self.lines = []
        for i in range(pred.shape[1]):
            l, = self.ax.plot(self.time_idx[0:self.window_size] + 1 + i, pred[0:self.window_size, i],
                              color=(0.298, 0.447, 0.690, self.alpha_list[i]))
            self.lines.append(l)
        self.gt, = self.ax.plot(self.time_idx[0:self.window_size], x[0:self.window_size], color='red')

        self.ax.set_xlabel('time index')
        self.ax.set_ylabel(self.dataset_type)

        self.slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
        self.slider = Slider(self.slider_ax, 'Index', 0, len(x) - self.window_size, valinit=self.start_index, valstep=1)

        self.slider.on_changed(self.update)

    def update(self, val):
        self.start_index = int(val)
        self.end_index = self.start_index + self.window_size

        min_y, max_y = np.inf, -np.inf
        for i, line in enumerate(self.lines):
            line.set_xdata(self.time_idx[self.start_index:self.end_index] + 1 + i)
            line.set_ydata(self.pred[self.start_index:self.end_index, i])
            min_test = np.min(self.pred[self.start_index:self.end_index, i])
            max_test = np.max(self.pred[self.start_index:self.end_index, i])
            if min_test < min_y:
                min_y = min_test
            if max_test > max_y:
                max_y = max_test

        self.gt.set_xdata(self.time_idx[self.start_index:self.end_index])
        self.gt.set_ydata(self.x[self.start_index:self.end_index])

        min_test, max_test = np.min(self.x[self.start_index:self.end_index]), np.max(
            self.x[self.start_index:self.end_index])
        if min_test < min_y:
            min_y = min_test
        if max_test > max_y:
            max_y = max_test

        xlim_offset = self.window_size * 0.05
        self.ax.set_xlim(
            [self.time_idx[self.start_index] - xlim_offset, self.time_idx[self.end_index - 1] + xlim_offset])

        ylim_offset = (max_y - min_y) * 0.05
        self.ax.set_ylim([min_y - ylim_offset, max_y + ylim_offset])
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


class IndividualInference:
    """A class that creates an interactive plot to visualize the predicted and ground truth values of a time series.
    Args:
        pred (ndarray): The predicted time series values.
            For example pred[t] gives the predictions for time steps [t+1, ..., t+T].
        gt (ndarray): The ground truth time series values.
            For example gt[t] = x[t-L, ..., t+T], where x is the ground truth.
        gt_t (ndarray): The time indices corresponding to each value in gt.
            For example gt_t[t] = [t-L, ..., t+T].
        pred_t (ndarray): The time indices corresponding to each value of pred.
            For example, pred_t[t] = [t+1, ..., t+T].
        dataset_type (str): The name of the dataset type - used for y-axis label.
    Example:
        building_index = 5
        dataset_type = 'price'
        expt_name = 'linear_L168_T48'
        predictor = Predictor(expt_name=expt_name, load=True)
        _, pred, gt, gt_t, pred_t, mse = predictor.test_individual(building_index, dataset_type)
        inference = IndividualInference(pred, gt, gt_t, pred_t, dataset_type)
        inference.show()
    Notes:
        The plot is interactive and allows the user to control the current window in the time series using a slider
        widget.
    """

    def __init__(self, pred, gt, gt_t, pred_t, dataset_type):
        self.gt_t = gt_t
        self.gt = gt
        self.pred_t = pred_t
        self.pred = pred
        self.dataset_type = dataset_type

        self.fig, self.ax = plt.subplots(figsize=[15, 5])
        plt.subplots_adjust(left=0.1, bottom=0.25)

        self.i = 0
        self.line_gt, = self.ax.plot(gt_t[self.i], gt[self.i], color='red')
        self.line_pred, = self.ax.plot(pred_t[self.i], pred[self.i], color='blue')

        self.line_v = self.ax.vlines(pred_t[self.i][0] - 1, self.ax.get_ylim()[0], self.ax.get_ylim()[1],
                                     colors='grey', linestyles='--', linewidth=1)

        self.ax.set_xlabel('time index')
        self.ax.set_ylabel(self.dataset_type)

        self.slider_ax = plt.axes([0.1, 0.1, 0.8, 0.05])
        self.slider = Slider(self.slider_ax, 'Index', 0, len(gt_t) - 1, valinit=0, valstep=1)

        self.slider.on_changed(self.update)

    def update(self, val):
        self.i = int(val)

        self.line_gt.set_xdata(self.gt_t[self.i])
        self.line_gt.set_ydata(self.gt[self.i])

        self.line_pred.set_xdata(self.pred_t[self.i])
        self.line_pred.set_ydata(self.pred[self.i])

        min_y, max_y = np.min(self.gt[self.i]), np.max(self.gt[self.i])
        min_test, max_test = np.min(self.pred[self.i]), np.max(self.pred[self.i])
        if min_test < min_y:
            min_y = min_test
        if max_test > max_y:
            max_y = max_test

        xlim_offset = len(self.gt_t[self.i]) * 0.05
        self.ax.set_xlim([self.gt_t[self.i][0] - xlim_offset, self.gt_t[self.i][-1] + xlim_offset])

        ylim_offset = (max_y - min_y) * 0.05
        self.ax.set_ylim([min_y - ylim_offset, max_y + ylim_offset])

        self.line_v.remove()
        self.line_v = self.ax.vlines(self.pred_t[self.i][0] - 1, self.ax.get_ylim()[0], self.ax.get_ylim()[1],
                                     colors='grey', linestyles='--', linewidth=1)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()
