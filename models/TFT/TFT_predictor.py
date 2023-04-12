"""Implementation of TFT predictor model class."""

import os
import glob
import json
from typing import Any, List, Dict, Union
import warnings

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss


class TFT_Predictor():

    def __init__(self,
        model_names: Union[List, Dict] = None,
        model_group_name = 'default',
        load: Union[str, bool] = 'group'
        ) -> None:

        self.model_types = ['load','solar','pricing','carbon']
        self.model_group_name = model_group_name
        self.model_group_path = os.path.join('models','TFT','resources','lightning_logs',model_group_name)

        if load in ['group','indiv']:
            assert os.path.exists(self.model_group_path), "Cannot load model group as logs do not exist!"

        # Perform model loading/construction.
        # ====================================================================
        # Construct dictionary of model names.
        if type(model_names) == list: # models provided as list of building indices
            assert all([type(item) == int for item in model_names])
            model_names = {
                'load': [f'load_{b}' for b in model_names],
                'solar': [f'solar_{b}' for b in model_names],
                'pricing': 'pricing',
                'carbon': 'carbon'
            }
        elif type(model_names) == dict: # model provided as dictionary of model names
            assert [key in list(model_names.keys) for key in self.model_types]
        elif not model_names: # model names not provided
            if not load: # initialise blank dict of model names to be filled later
                model_names = {
                    'load': [],
                    'solar': [],
                    'pricing': None,
                    'carbon': None
                }
            elif load == 'group': # get model names for given `model_group_name`
                for model_type in self.model_types:
                    assert os.path.exists(os.path.join(self.model_group_path),model_type), f"`{model_type}` model logs sub-dir does not exist."
                model_names = {
                    'load': [f.path for f in os.scandir(os.path.join(self.model_group_path,'load')) if f.is_dir()],
                    'solar': [f.path for f in os.scandir(os.path.join(self.model_group_path,'solar')) if f.is_dir()],
                    'pricing': 'pricing',
                    'carbon': 'carbon'
                }
            else:
                raise ValueError("Cannot use load option `indiv` if model names are not specified.")
        else:
            raise ValueError("`models` argument must be: list of building indices, dict of model names, or None.")

        self.model_names = model_names

        # Load or initialise models.
        if load in ['group','indiv']:
            self.load()
        elif not load: # initialise models dict to be filled later
            self.models = {
                'load': {},
                'solar': {},
                'pricing': {},
                'carbon': {}
            }
            if os.path.exists(self.model_group_path):
                warnings.warn(f"Warning: A log directory for this model group name ({model_group_name}) already exists but you are not loading it.")
            else:
                os.makedirs(os.path.realpath(self.model_group_path))
        else:
            raise ValueError("`load` argument must be: 'group', 'indiv', or False")


        # Perform prediction setup.
        # ====================================================================
        # define indices of variables in observations
        self.load_obs_index = 20
        self.solar_obs_index = 21
        self.pricing_obs_index = 24
        self.carbon_obs_index = 19

        # initialise observations buffer
        self.buffer = {
            'load': [],
            'solar': [],
            'pricing': [],
            'carbon': []
        }


    def load(self) -> None:

        self.models = {
            'load': {model_name: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,model_name)) for model_name in self.model_names['load']},
            'solar': {model_name: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,model_name)) for model_name in self.model_names['solar']},
            'pricing': {self.model_names['pricing']: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,self.model_names['pricing']))},
            'carbon': {self.model_names['carbon']: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,self.model_names['carbon']))}
        }
        # Note: order of models in dictionary defines models used for prediction at each index in load and solar arrays


    def load_TFT_from_model_dir_path(self, model_path) -> TemporalFusionTransformer:

        # Note: best_model.json contains a dict with one key `rel_path`, whose values is a list
        # of the relative path (strings) to the checkpoint file of the best version of that model

        load_path_file = 'best_model.json'
        json_path = os.path.exists(model_path,load_path_file)
        assert os.path.exists(json_path), "JSON indicating checkpoint file to load `{json_path}` does not exist."

        with open(json_path,'r') as json_file:
            best_model_chkpt = json.load(json_file)['rel_path']
        best_model_path = os.path.join(model_path, *best_model_chkpt)

        return TemporalFusionTransformer.load_from_checkpoint(best_model_path)


    def format_CityLearn_datasets(
        self,
        CityLearn_dataset_dirpaths,
        model_type,
        building_index: int = None,
        max_encoder_length=72,
        max_prediction_length=48
        ) -> List[TimeSeriesDataSet]:
        # format appropriate TimeSeriesDataSet from CityLearn dataset dir(s)
        # use construction from_dataset for datasets after first

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."

        building_fname_pattern = 'UCam_Building*.csv'

        if (model_type in ['load','solar']) and not building_index:
            raise ValueError(f"Must supply `building_ID` to construct {model_type} dataset from CityLearn data.")

        # Specify column name variables.
        time_id_col_name = 'time_idx'
        load_col_name = 'Equipment Electric Power [kWh]'
        temp_col_name = 'Outdoor Drybulb Temperature [C]'
        solar_col_name = 'Solar Generation Power [kW]'
        dif_irad_col_name = 'Diffuse Solar Radiation [W/m2]'
        dir_irad_col_name = 'Direct Solar Radiation [W/m2]'
        pricing_col_name = 'Electricity Pricing [£]'
        carbon_col_name = 'Carbon Intensity [kg_CO2/kWh]'

        # Set default kwargs.
        if model_type == 'load':
            target = load_col_name
            ts_name = 'l'
            time_varying_unknown_reals = [load_col_name,temp_col_name]
        elif model_type == 'solar':
            target = solar_col_name
            ts_name = 's'
            time_varying_unknown_reals = [solar_col_name,dif_irad_col_name,dir_irad_col_name]
        elif model_type == 'pricing':
            target = pricing_col_name
            ts_name = 'p'
            time_varying_unknown_reals = [pricing_col_name]
        elif model_type == 'carbon':
            target = carbon_col_name
            ts_name = 'c'
            time_varying_unknown_reals = [carbon_col_name]

        time_varying_known_categoricals = ['Month','Hour','Day Type','Daylight Savings Status']

        def reformat_df(df,ts_name,tv_cats):
            df = df.rename_axis('time_idx').reset_index() # create column of indices to pass as time_idx to TimeSeriesDataSet - we have no missing values
            df['ts_id'] = ts_name # create column with ID of timeseries (constant as only single timeseries)
            for col in tv_cats:
                df[col] = df[col].astype(str) # convert to strs to use as categoric covariates
            return df

        ts_datasets = []

        for i,dirpath in enumerate(CityLearn_dataset_dirpaths):

            # Grab data from CityLearn files & environment.
            env = CityLearnEnv(os.path.join(dirpath,'schema.json'))
            # construct base dataframe with time info (known categoricals)
            first_building_file_path = glob.glob(os.path.join(dirpath,building_fname_pattern))[0]
            data_df = pd.read_csv(first_building_file_path,usecols=time_varying_known_categoricals)
            data_df = reformat_df(data_df, ts_name, time_varying_known_categoricals)

            # add type specific data to df, taking data from environment
            if model_type == 'load':
                data_df[load_col_name] = env.buildings[building_index].energy_simulation.non_shiftable_load
                data_df[temp_col_name] = env.buildings[building_index].weather.outdoor_dry_bulb_temperature
            elif model_type == 'solar':
                data_df[solar_col_name] = env.buildings[building_index].energy_simulation.solar_generation
                data_df[dif_irad_col_name] = env.buildings[building_index].weather.diffuse_solar_irradiance
                data_df[dir_irad_col_name] = env.buildings[building_index].weather.direct_solar_irradiance
            elif model_type == 'pricing':
                data_df[pricing_col_name] = env.buildings[0].pricing.electricity_pricing
            elif model_type == 'carbon':
                data_df[carbon_col_name] = env.buildings[0].carbon_intensity.carbon_intensity

            if i == 0:
                timeseries_dataset = TimeSeriesDataSet(
                    data_df,
                    time_idx=time_id_col_name,  # column name of time of observation
                    target=target,  # column name of target to predict
                    group_ids=['ts_id'],  # column name(s) for timeseries IDs (static as only 1 timeseries used)
                    max_encoder_length=max_encoder_length,  # how much history to use
                    max_prediction_length=max_prediction_length,  # how far to predict into future
                    # covariates static for a timeseries ID - ignore for the moment
                    #static_categoricals=[ ... ],
                    #static_reals=[ ... ],
                    # covariates known and unknown in the future to inform prediction
                    time_varying_known_categoricals=time_varying_known_categoricals,
                    #time_varying_known_reals=[ ... ],
                    #time_varying_unknown_categoricals=[ ... ],
                    time_varying_unknown_reals=time_varying_unknown_reals
                )
            else:
                timeseries_dataset = TimeSeriesDataSet.from_dataset(ts_datasets[0], data_df)

            ts_datasets.append(timeseries_dataset)

        return ts_datasets



    def new_model(self, model_name: str, model_type: str, train_dataset: TimeSeriesDataSet, **kwargs) -> TemporalFusionTransformer:

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."
        assert type(train_dataset) == TimeSeriesDataSet, "`train_dataset` must be a pytorch_forecasting.TimeSeriesDataSet object."

        model_path = os.path.join(self.model_group_path,model_name)

        if os.path.exists:
            warnings.warn(f"Warning: A logs directory already exists for the model name `{model_name}`. By continuing you will overwrite this model.")
            if input("Are you sure you want to overwrite this model? [y/n]") not in ['y','Y']:
                print("Aborting model creation.")
                return

        # Set default kwargs.
        # architecture hyperparameters
        if not kwargs['hidden_size']: kwargs['hidden_size'] = 48,
        if not kwargs['attention_head_size']: kwargs['attention_head_size'] = 4,
        if not kwargs['dropout']: kwargs['dropout'] = 0.1,
        if not kwargs['hidden_continuous_size']: kwargs['hidden_continuous_size'] = 16,
        # loss metric to optimize
        if not kwargs['loss']: kwargs['loss'] = QuantileLoss(),
        # set optimizer
        if not kwargs['optimizer']: kwargs['optimizer'] = 'adam',
        # optimizer parameters
        if not kwargs['reduce_on_plateau_patience']: kwargs['reduce_on_plateau_patience'] = 5

        # initialise TFT model from train_dataset specification
        tft = TemporalFusionTransformer.from_dataset(train_dataset,**kwargs)

        # set model
        if model_type in ['load','solar']:
            self.model_names[model_type].append(model_name)
        else:
            warnings.warn(f"Warning: f{model_type} model replaced.")
            self.model_names[model_type] = model_name
        self.models[model_type][model_name] = tft

        os.makedirs(os.path.realpath(model_path))

        return tft


    def train(self, model_name: str, model_type: str, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, **kwargs) -> None:

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."
        assert model_name in self.model_names[model_type], f"Model {model_name} of type {model_type} not loaded into predictor."

        # continue training is checkpoint file available
        load_path_file = 'best_model.json'
        if os.path.exists(os.path.join(self.model_group_path,model_name,load_path_file)):
            json_path = os.path.join(self.model_group_path,model_name,load_path_file)
            with open(json_path,'r') as json_file:
                best_model_chkpt = json.load(json_file)['rel_path']
            best_checkpoint_path = os.path.join(self.model_group_path,model_name,*best_model_chkpt)
        else:
            best_checkpoint_path = None

        # Get model.
        model = self.models[model_type][model_name]

        # Convert datasets to dataloaders for training.
        batch_size = 128
        n_workers = 4
        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=n_workers)
        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=n_workers)

        # Create PyTorch Lightning Trainer with early stopping.
        # TODO: allow training kwargs to be taken in as method kwargs
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=5, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        tb_logger = TensorBoardLogger(save_dir=self.model_group_path, name=model_name)
        trainer = pl.Trainer(
            deterministic=True,
            accelerator='cpu',
            max_epochs=100,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=500,  # batches per epoch
            callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
            logger=tb_logger
        )

        # Tune learning rate.
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloader, val_dataloader, min_lr=1e-6, max_lr=1e-1)
        lr = lr_finder.suggestion()
        model.learning_rate = lr

        # Do training.
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=best_checkpoint_path)

        # Save best_model.json to log directory.
        best_model_path = trainer.checkpoint_callback.best_model_path
        path_components = []
        current_level = None
        remaining_path = best_model_path

        while current_level != model_name:
            remaining_path, current_level = os.path.split(remaining_path)
            path_components.insert(0, current_level)
        best_model_path_dict = {'rel_path':path_components}

        with open(json_path,'w') as json_file:
            json.dump(best_model_path_dict, json_file)


    def compute_forecast(self, observations):
        pass