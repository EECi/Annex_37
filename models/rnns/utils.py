import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import Dataset
import lightning.pytorch as pl
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet
from typing import Any, List, Dict, Union
import pickle
# filer warnings: mostly object reload & numpy deprecation warnings
import warnings

warnings.filterwarnings(action='ignore', module=r'pytorch_forecasting')


def model_finder(model_name, mparam):
    if model_name == 'vanilla':
        return Vanilla(**mparam)
    elif model_name == 'lstm':  # model_name = 'lstm'
        return LSTMmodel(**mparam)
    elif model_name == 'tsfm':  # model_name = 'lstm'
        return TransformerModel(**mparam)
    elif model_name == 'gru':  # model_name = 'lstm'
        return GRUmodel(**mparam)
    elif model_name == 'DeepAR':  # model_name = 'lstm'
        return DeepAR(**mparam)
    elif model_name == 'TiDe':  # model_name = 'lstm'
        return TiDe(**mparam)

    else:
        return None


class LSTMmodel(pl.LightningModule):
    def __init__(self, L, T, layers=128, hidden_size=64, n_features=15, learning_rate=1e-3):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_size, batch_first=True)
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_size, layers),
            nn.ReLU()
        )
        self.linear = nn.Linear(layers, self.output_dim)
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.n_features, 1), stride=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.hidden_layers(x)
        x = self.linear(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TransformerModel(pl.LightningModule):

    def __init__(self, L, T, layers=128, hidden_size=64, n_features=15, learning_rate=1e-3):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(self.input_dim, self.hidden_size)
        self.transformer = nn.Transformer(self.hidden_size, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_size, layers),
            nn.ReLU()
        )
        self.linear = nn.Linear(layers, self.output_dim)
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.n_features, 1), stride=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.permute(1, 0, 2), x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.hidden_layers(x)
        x = self.linear(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class GRUmodel(pl.LightningModule):
    def __init__(self, L, T, layers=128, hidden_size=64, n_features=15, learning_rate=1e-3):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=hidden_size, batch_first=True)
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_size, layers),
            nn.ReLU()
        )
        self.linear = nn.Linear(layers, self.output_dim)
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.n_features, 1), stride=1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.hidden_layers(x)
        x = self.linear(x)
        x = self.avg_pool(x)
        x = x.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class DeepAR(pl.LightningModule):
    def __init__(self, L, T, layers=128, hidden_size=64, n_features=15, learning_rate=1e-3, num_layers=2, dropout=0.5,
                 output_distribution="Gaussian"):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_size, layers),
            nn.ReLU()
        )
        self.output_distribution = output_distribution
        if self.output_distribution == "Gaussian":
            self.loc = nn.Linear(layers, self.output_dim)
            self.scale = nn.Linear(layers, self.output_dim)
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.n_features, 1), stride=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.hidden_layers(x)
        if self.output_distribution == "Gaussian":
            loc = self.loc(x)
            loc  = self.avg_pool(loc)
            loc  = loc.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
            scale = self.scale(x)
            scale = self.avg_pool(scale)
            scale = scale.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
            scale = torch.exp(scale)  # Ensure the scale is positive
            return loc, scale #discarding the uncertainty information captured by the scale tensor
        else:
            raise NotImplementedError("Output distribution not supported")

    def training_step(self, batch, batch_idx):
        x, y = batch

        loc, scale = self(x)
        if self.output_distribution == "Gaussian":
            loss = torch.mean(torch.sum(0.5 * torch.log(scale ** 2) + 0.5 * ((y - loc) / scale) ** 2, dim=1))
        else:
            raise NotImplementedError("Loss function for output distribution not supported")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loc, scale = self(x)
        if self.output_distribution == "Gaussian":
            loss = torch.mean(torch.sum(0.5 * torch.log(scale ** 2) + 0.5 * ((y - loc) / scale) ** 2, dim=1))
        else:
            raise NotImplementedError("Loss function for output distribution not supported")
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class TiDe(pl.LightningModule):
    def __init__(self, L, T, hidden_size=64, num_layers=2, dropout=0.5, learning_rate=1e-3,n_features=15):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.encoder = nn.LSTM(input_size=L, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                               dropout=dropout)
        self.debias = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                               dropout=dropout)
        self.loc = nn.Linear(hidden_size, T)
        self.scale = nn.Linear(hidden_size, T)
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.n_features, 1), stride=1)

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.debias(x)
        x, _ = self.decoder(x)
        loc = self.loc(x)
        loc = self.avg_pool(loc)
        loc = loc.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
        scale = self.scale(x)
        scale = self.avg_pool(scale)
        scale = scale.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
        scale = torch.exp(scale)  # Ensure the scale is positive
        return loc, scale # discarding the uncertainty information captured by the scale tensor

    def training_step(self, batch, batch_idx):
        x, y = batch
        print("y shape:", y.shape)  # Print y shape
        loc, scale = self(x)
        loss = torch.mean(torch.sum(0.5 * torch.log(scale ** 2) + 0.5 * ((y - loc) / scale) ** 2, dim=1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loc, scale = self(x)
        loss = torch.mean(torch.sum(0.5 * torch.log(scale ** 2) + 0.5 * ((y - loc) / scale) ** 2, dim=1))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class Vanilla(pl.LightningModule):
    def __init__(self, L, T, layers=(128, 256), learning_rate=1e-3, feature=15):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate
        self.feature = feature
        if self.h:
            self.input_layer = torch.nn.Linear(self.input_dim, layers[0])
            self.hidden_layers = torch.nn.Sequential()
            for i in range(len(layers) - 1):
                self.hidden_layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
            self.output_layer = torch.nn.Linear(layers[-1], self.output_dim)
        else:
            self.output_layer = torch.nn.Linear(self.input_dim, self.output_dim)

        self.avg_pool = nn.AvgPool2d(kernel_size=(feature, 1), stride=1)

    def forward(self, x):
        if self.h:
            x = torch.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = torch.relu(layer(x))
        x = self.output_layer(x)
        x = x.unsqueeze(1)
        y = self.avg_pool(x)
        y = y.squeeze(2)  # Remove the pooled dimension (batch_size, channel, 1, feature)
        return y

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
        data_categoricals = building[['Month', 'Hour', 'Day Type', 'Daylight Savings Status']].copy()
        data_numerical = pd.concat([load, solar, price, carbon, weather], axis=1)
        data_numerical.columns = ['load', 'solar', 'price', 'carbon', 'Tem', 'Hum', 'DifSolar', 'DirSolar']
        # Cyclical encoding for month, day, and hour weather.columns
        data_categoricals.loc[:, 'month_sin'] = np.sin(2 * np.pi * data_categoricals.loc[:, 'Month'] / 12)
        data_categoricals.loc[:, 'month_cos'] = np.cos(2 * np.pi * data_categoricals.loc[:, 'Month'] / 12)
        data_categoricals.loc[:, 'day_sin'] = np.sin(2 * np.pi * data_categoricals.loc[:, 'Day Type'] / 31)
        data_categoricals.loc[:, 'day_cos'] = np.cos(2 * np.pi * data_categoricals.loc[:, 'Day Type'] / 31)
        data_categoricals.loc[:, 'hour_sin'] = np.sin(2 * np.pi * data_categoricals.loc[:, 'Hour'] / 24)
        data_categoricals.loc[:, 'hour_cos'] = np.cos(2 * np.pi * data_categoricals.loc[:, 'Hour'] / 24)
        data_categoricals.loc[:, 'daylight'] = data_categoricals.loc[:, 'Daylight Savings Status'].astype(str)
        data_categoricals = data_categoricals.drop(['Month', 'Hour', 'Day Type', 'Daylight Savings Status'], axis=1)
        data = pd.concat([data_numerical, data_categoricals],
                         axis=1)  # data.shape  data[dataset_type].values.reshape(-1, 1).shape
        data.index.name = 'time_index'
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaled_data_x = scaler_x.fit_transform(data)
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaled_data_y = scaler_y.fit_transform(data[self.dataset_type].values.reshape(-1, 1))
        dataset_params_dir = os.path.join("models", "rnns", "resources", self.expt_name, self.key)
        os.makedirs(dataset_params_dir, exist_ok=True)
        dataset_params_path_x = os.path.join(
            f'{os.path.join("models", "rnns", "resources", self.expt_name, self.key, "scalar_x.pkl")}')
        with open(dataset_params_path_x, 'wb') as pkl_file:
            pickle.dump(scaler_x, pkl_file)
        dataset_params_path_y = os.path.join(
            f'{os.path.join("models", "rnns", "resources", self.expt_name, self.key, "scalar_y.pkl")}')
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
        self.x = np.array(self.x, dtype=np.float32)
        self.x = np.transpose(self.x, (0, 2, 1))
        self.y = np.array(self.y, dtype=np.float32)
        self.y = np.transpose(self.y, (0, 2, 1))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
