import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import Dataset
import lightning.pytorch as pl


def model_finder(model_name, mparam):
    """Instantiates and returns an instance of the specified model.
    Args:
        model_name (str): The name of the model to be created.
        mparam (dict): A dictionary of model parameters.
    Returns:
        The newly created model instance, or None if the specified model_name is not supported.
    Example:
        model_name = 'vanilla',
        mparam = {'L': 144, 'T': 48, 'layers': []}
        model = model_finder(model_name, mparam)
    """
    if model_name == 'vanilla':
        return Vanilla(**mparam)
    else:
        return None


class Vanilla(pl.LightningModule):
    """A PyTorch Lightning module for a vanilla neural network.
    Args:
        L (int): Input dimension.
        T (int): Output dimension.
        layers (tuple, optional): Tuple of hidden layer sizes. Defaults to (128, 256).
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
    Methods:
        forward(x): Defines the forward pass of the neural network.
        training_step(batch, batch_idx): Performs a training step on a batch of data.
        validation_step(batch, batch_idx): Performs a validation step on a batch of data.
        configure_optimizers(): Configures the optimizer used for training.
    Notes:
        The neural network is an MLP with the following dimensions: (input, layers[0], layers[1], ..., output).
    """

    def __init__(self, L, T, layers=(128, 256), learning_rate=1e-3):
        super().__init__()
        self.input_dim = L
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate

        if self.h:
            self.input_layer = torch.nn.Linear(self.input_dim, layers[0])
            self.hidden_layers = torch.nn.Sequential()
            for i in range(len(layers) - 1):
                self.hidden_layers.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.output_layer = torch.nn.Linear(layers[-1], self.output_dim)
        else:
            self.output_layer = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        if self.h:
            x = torch.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x

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
    """Represents a dataset for energy consumption (load), solar generation (solar), pricing (price), and carbon intensity
    (carbon).
    Args:
        building_index (int): The index of the building to use.
        L (int): The length of the input sequence.
        T (int): length of planning horizon (number of time instances into the future to forecast).
        dataset_type (str):The type of the dataset to use ('load', 'solar', 'price', or 'carbon').
        version (str): The version of the dataset to use ('train', 'valid' or 'test').
        outside_module (bool): Specifies that the Dataset is used outside the dms module, for instance when
            used in evaluate.py via the predictor.
    Example:
        data = Data(building_index=5, L=48, T=24, dataset_type='load', version='train')
    """

    def __init__(self, building_index=5, L=48, T=24, dataset_type='load', version='train', outside_module=False):
        super().__init__()
        self.dataset_type = dataset_type
        self.type2idx = {'load': 0, 'solar': 1, 'price': 2, 'carbon': 3}
        self.type_idx = self.type2idx[dataset_type]

        dataset_dir = os.path.join('data', 'example')
        if not outside_module:
            dataset_dir = os.path.join('..', '..', dataset_dir)
        building = pd.read_csv(os.path.join(dataset_dir, version, f'UCam_Building_{building_index}.csv'))
        load = building['Equipment Electric Power [kWh]']
        solar = building['Solar Generation [W/kW]']
        price = pd.read_csv(os.path.join(dataset_dir, version, 'pricing.csv'))['Electricity Pricing [£/kWh]']
        carbon = pd.read_csv(os.path.join(dataset_dir, version, 'carbon_intensity.csv'))
        data = pd.concat([load, solar, price, carbon], axis=1)
        data.columns = ['load', 'solar', 'price', 'carbon']
        data.index.name = 'time_index'

        self.x = []
        self.y = []
        for i in range(len(data)):
            if i == len(data) - L - T + 1:
                break
            else:
                self.x.append([data['load'].iloc[i:i + L],
                               data['solar'].iloc[i:i + L],
                               data['price'].iloc[i:i + L],
                               data['carbon'].iloc[i:i + L]])
                self.y.append([data['load'].iloc[i + L:i + L + T],
                               data['solar'].iloc[i + L:i + L + T],
                               data['price'].iloc[i + L:i + L + T],
                               data['carbon'].iloc[i + L:i + L + T]])
        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx][self.type_idx], self.y[idx][self.type_idx]


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
        self.slider = Slider(self.slider_ax, 'Index', 0, len(x)-self.window_size, valinit=self.start_index, valstep=1)

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

        min_test, max_test = np.min(self.x[self.start_index:self.end_index]), np.max(self.x[self.start_index:self.end_index])
        if min_test < min_y:
            min_y = min_test
        if max_test > max_y:
            max_y = max_test

        xlim_offset = self.window_size * 0.05
        self.ax.set_xlim([self.time_idx[self.start_index] - xlim_offset, self.time_idx[self.end_index - 1] + xlim_offset])

        ylim_offset = (max_y - min_y) * 0.05
        self.ax.set_ylim([min_y-ylim_offset, max_y + ylim_offset])
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