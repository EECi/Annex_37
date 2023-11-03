import os
from typing import List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import Dataset


class Data(Dataset):
    """Represents a dataset for: energy consumption (load), solar generation (solar), grid electricity
    pricing (price), and carbon intensity (carbon); for a particular building.

    Args:
        building_index (int): The index of the building to use.
        L (int): The length of the input sequence.
        T (int): length of planning horizon (number of time instances into the future to forecast).
        dataset_type (str):The type of the dataset to use ('load', 'solar', 'price', or 'carbon').
        version (str): The version of the dataset to use ('train', 'valid' or 'test').
    Example:
        data = Data(building_index=5, L=48, T=24, version='train')
    """

    def __init__(self, building_index: int, L: int, T: int = 48, control_inputs: List = None,
                 version='train', dataset_dir=os.path.join('data', 'analysis')):
        super().__init__()

        self.control_inputs = control_inputs

        # load data from CSVs
        building = pd.read_csv(os.path.join(dataset_dir, version, f'UCam_Building_{building_index}.csv'))
        data_variables = {
            'load': building['Equipment Electric Power [kWh]'],
            'solar': building['Solar Generation [W/kW]'],
            'price': pd.read_csv(os.path.join(dataset_dir, version, 'pricing.csv'))['Electricity Pricing [£/kWh]'],
            'carbon': pd.read_csv(os.path.join(dataset_dir, version, 'carbon_intensity.csv')),
            'diff_solar': pd.read_csv(os.path.join(dataset_dir, version, 'weather.csv'))['Diffuse Solar Radiation [W/m2]'],
            'dir_solar': pd.read_csv(os.path.join(dataset_dir, version, 'weather.csv'))['Direct Solar Radiation [W/m2]']
        }

        # load collected data into dataframe
        self.columns = ['load', 'solar', 'price', 'carbon']
        if control_inputs != None:
            self.columns += control_inputs
        data_df = pd.concat([data_variables[col] for col in self.columns], axis=1)
        data_df.columns = self.columns
        data_df.index.name = 'time_index'
        data_df = data_df.fillna(0)
        self.data_df = data_df
        self.data = np.array(data_df, dtype=np.float32).T

    def __len__(self):
        return len(self.data.shape[0])
