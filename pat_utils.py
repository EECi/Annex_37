import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import lightning.pytorch as pl


class Data(Dataset):
    def __init__(self, building_index=5, L=48, T=24, version='train', dataset_type='load'):
        super().__init__()
        self.dataset_type = dataset_type
        self.type2idx = {'load': 0, 'solar': 1, 'price': 2, 'carbon': 3}
        self.type_idx = self.type2idx[dataset_type]

        dataset_dir = os.path.join('data', 'example')
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


def model_finder(config, mparam):
    if config['model'] == 'vanilla':
        return Vanilla(**mparam)
    elif config['model'] == 'normalise':
        return NormaliseModel(**mparam)
    elif config['model'] == 'normalise_feature':
        return NormaliseFeatureModel(**mparam)
    elif config['model'] == 'avg_kernel':
        return AVGKernelModel(**mparam)
    else:
        return None


def get_expt_name(config, mparam):
    keys_c = list(config.keys())
    keys_m = list(mparam.keys())
    keys = keys_c + keys_m
    keys.sort()
    combined_dict = config.copy()
    combined_dict.update(mparam)
    out_str = ''
    for key in keys:
        value = combined_dict[key]
        if type(value) == list:
            out_str += key
            out_str += str(value).replace('[', '').replace(']', '').replace(', ', '_') + '_'
        elif type(value) == str:
            out_str += str(value) + '_'
        elif type(value) == bool:
            out_str += key
            out_str += str(value)[0].lower() + '_'
        else:
            out_str += key
            out_str += str(value) + '_'
    return out_str[:-1]


class Vanilla(pl.LightningModule):
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


class NormaliseModel(pl.LightningModule):
    def __init__(self, L, T, layers=(128, 256), mean=True, std=True,
                 learning_rate=1e-3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.input_dim = L
        self.output_dim = T
        self.learning_rate = learning_rate
        self.h = layers

        if self.h:
            self.input_layer = torch.nn.Linear(self.input_dim, layers[0])
            self.hidden_layers = torch.nn.Sequential()
            for i in range(len(layers) - 1):
                self.hidden_layers.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.output_layer = torch.nn.Linear(layers[-1], self.output_dim)
        else:
            self.output_layer = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        std = torch.sqrt(var + 1e-10)

        if self.mean:
            x -= mean
        if self.std:
            x /= std

        if self.h:
            x = torch.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = torch.relu(layer(x))
        x = self.output_layer(x)

        if self.std:
            x *= std
        if self.mean:
            x += mean
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


class NormaliseFeatureModel(pl.LightningModule):
    def __init__(self, L, T, layers=(128, 256), mean=True, std=True,
                 learning_rate=1e-3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.input_dim = L
        self.output_dim = T
        self.learning_rate = learning_rate
        self.h = layers

        extra_features = 0
        if self.mean:
            extra_features += 1
        if self.std:
            extra_features += 1
        self.input_dim += extra_features

        if self.h:
            self.input_layer = torch.nn.Linear(self.input_dim, layers[0])
            self.hidden_layers = torch.nn.Sequential()
            for i in range(len(layers) - 1):
                self.hidden_layers.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.output_layer = torch.nn.Linear(layers[-1], self.output_dim)
        else:
            self.output_layer = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True)
        std = torch.sqrt(var + 1e-10)

        if self.mean:
            x -= mean
            x = torch.cat((x, mean), dim=1)
        if self.std:
            x /= std
            x = torch.cat((x, std), dim=1)

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


class AVGKernelModel(pl.LightningModule):
    def __init__(self, L, T, layers=(128, 256), k=41, learning_rate=1e-3):
        super().__init__()
        assert (k % 2) == 1, "kernel size must be an odd integer"

        self.kernel_size = k
        self.input_dim = L*2
        self.output_dim = T
        self.h = layers
        self.learning_rate = learning_rate

        self.avg = torch.nn.AvgPool1d(kernel_size=k, stride=1, padding=(k-1)//2)

        if self.h:
            self.input_layer = torch.nn.Linear(self.input_dim, layers[0])
            self.hidden_layers = torch.nn.Sequential()
            for i in range(len(layers) - 1):
                self.hidden_layers.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.output_layer = torch.nn.Linear(layers[-1], self.output_dim)
        else:
            self.output_layer = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x_mean = self.avg(x)
        res = x - x_mean
        x = torch.cat([x_mean, res], dim=1)

        # todo: test mean of entire input
        # mean = x.mean(dim=1, keepdim=True)
        # res = x - mean
        # mean = mean.repeat([1, x.shape[1]])
        # x = torch.cat([res, mean], dim=1)

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
