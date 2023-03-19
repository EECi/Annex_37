import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import lightning.pytorch as pl


class Data(Dataset):
    def __init__(self, building_index=5, L=48, T=24, version='train'):
        super().__init__()
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
        return self.x[idx][0], self.y[idx][0]   # todo: just return load: 1d for now


class Model(pl.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        # self.fc1 = torch.nn.Linear(input_dim, output_dim)
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc1(x)
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