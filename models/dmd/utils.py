import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import Dataset


class Data(Dataset):
    """Represents a dataset for energy consumption (load), solar generation (solar), pricing (price), and carbon intensity
    (carbon).

    Args:
        building_index (int): The index of the building to use.
        L (int): The length of the input sequence.
        T (int): length of planning horizon (number of time instances into the future to forecast).
        dataset_type (str):The type of the dataset to use ('load', 'solar', 'price', or 'carbon').
        version (str): The version of the dataset to use ('train', 'valid' or 'test').

    Example:
        data = Data(building_index=5, L=48, T=24, dataset_type='load', version='train')
    """

    def __init__(self, building_index=11, L=None, T=None, dataset_type='load', version='train', control_inputs=None):
        super().__init__()
        self.dataset_type = dataset_type
        self.type2idx = {'load': 0, 'solar': 1, 'price': 2, 'carbon': 3, }
        self.type_idx = self.type2idx[dataset_type]

        # self.type2idy = {'diff_solar': 0, 'dir_solar': 1}
        # self.type_idy = self.type2idy['weather']

        self.control_inputs = control_inputs

        dataset_dir = os.path.join('data', 'example')
        building = pd.read_csv(os.path.join(dataset_dir, version, f'UCam_Building_{building_index}.csv'))

        load = building['Equipment Electric Power [kWh]']
        solar = building['Solar Generation [W/kW]']
        price = pd.read_csv(os.path.join(dataset_dir, version, 'pricing.csv'))['Electricity Pricing [£/kWh]']
        carbon = pd.read_csv(os.path.join(dataset_dir, version, 'carbon_intensity.csv'))
        diff_solar = pd.read_csv(os.path.join(dataset_dir, version, 'weather.csv'))[
            'Diffuse Solar Radiation [W/m2]']
        dir_solar = pd.read_csv(os.path.join(dataset_dir, version, 'weather.csv'))['Direct Solar Radiation [W/m2]']

        if control_inputs!= None:
            # c_input = pd.read_csv(os.path.join(dataset_dir, version, 'weather.csv'))[control_inputs]
            """
            ### attempt to automate/generalise input of any control strings 
            control_data = []
            # for inp in control_inputs:
            #     control_data.append(exec("%s = %d" % (inp,diff)))
            # data = pd.concat([load, solar, price, carbon], axis=1)
            # data += control_data
            """
            data = pd.concat([load, solar, price, carbon, diff_solar, dir_solar], axis=1)
            data.columns = ['load', 'solar', 'price', 'carbon'] + control_inputs


        else:
            data = pd.concat([load, solar, price, carbon], axis=1)
            data.columns = ['load', 'solar', 'price', 'carbon']

        data.index.name = 'time_index'
        self.data = data.fillna(0)

        self.x = []
        self.y = []
        for i in range(len(data)):
            if i == len(data) - L + 1:
                break
            else:
                """
                d = [data['load'].iloc[i:i + L],
                               data['solar'].iloc[i:i + L],
                               data['price'].iloc[i:i + L],
                               data['carbon'].iloc[i:i + L]]
                if control_inputs != None:
                    d.append(data['diff_solar'].iloc[i:i + L])
                    d.append(data['dir_solar'].iloc[i:i + L])

                self.x.append(d)
                """
                self.x.append([data['load'].iloc[i:i + L],
                               data['solar'].iloc[i:i + L],
                               data['price'].iloc[i:i + L],
                               data['carbon'].iloc[i:i + L]])
                #
                if control_inputs!= None:
                    self.y.append([data['diff_solar'].iloc[i:i + L],
                                   data['dir_solar'].iloc[i:i + L]])

                # self.y.append([data['load'].iloc[i + L:i + L + T],
                #                data['solar'].iloc[i + L:i + L + T],
                #                data['price'].iloc[i + L:i + L + T],
                #                data['carbon'].iloc[i + L:i + L + T]])
                #                # data['diff_solar'].iloc[i:i + L],
                #                # data['dir_solar'].iloc[i:i + L]])

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def get_df (self):
        return self.data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.control_inputs != None:
            return self.x[idx][self.type_idx]
        else: return self.x[idx][self.type_idx]

