"""Implementation of TFT predictor model class."""

import os
import glob
import shutil
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
        model_group_name = 'default',
        L: int = 72,
        T: int = 48,
        model_names: Union[List, Dict] = None,
        load: Union[str, bool] = 'group'
        ) -> None:

        self.model_types = ['load','solar','pricing','carbon']
        self.model_group_name = model_group_name
        self.model_group_path = os.path.join('models','TFT','resources','model_logs',model_group_name)

        if load in ['group','indiv']:
            assert os.path.exists(self.model_group_path), "Cannot load model group as logs do not exist!"
        else:
            assert all([(type(val) == int) and (val > 0) for val in [L,T]])

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
                for model_type in ['pricing','carbon']:
                    if len(list(os.scandir(os.path.join(self.model_group_path,'solar')))) > 1:
                        warnings.warn(f"Warning: More than 1 {model_type} model available. 1st selected in group load.")
                model_names = {
                    'load': [os.path.split(f.path)[-1] for f in os.scandir(os.path.join(self.model_group_path,'load')) if f.is_dir()],
                    'solar': [os.path.split(f.path)[-1] for f in os.scandir(os.path.join(self.model_group_path,'solar')) if f.is_dir()],
                    'pricing': os.path.split(list(os.scandir(os.path.join(self.model_group_path,'pricing')))[0])[-1],
                    'carbon': os.path.split(list(os.scandir(os.path.join(self.model_group_path,'carbon')))[0])[-1]
                }
            else:
                raise ValueError("Cannot use load option `indiv` if model names are not specified.")
        else:
            raise ValueError("`models` argument must be: list of building indices, dict of model names, or None.")

        self.model_names = model_names

        # Load or initialise models.
        # NOTE: order of models in dictionary defines models used for prediction at each index in load and solar arrays (i.e. for eac building)

        group_mparams_path = os.path.join(self.model_group_path,'model_group_params.json')
        if load in ['group','indiv']:
            self.load()

            assert os.path.exists(group_mparams_path)
            with open(group_mparams_path,'r') as json_file:
                group_mparams = json.load(json_file)['model_group_parameters']
            self.L = group_mparams['L']
            self.T = group_mparams['T']

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
            
            if os.path.exists(group_mparams_path): warnings.warn("Warning: Model group metadata overwritten.")
            with open(group_mparams_path,'w') as json_file:
                json.dump({'model_group_parameters':{'L':L,'T':T}}, json_file)
            self.L = L
            self.T = T

        else:
            raise ValueError("`load` argument must be: 'group', 'indiv', or False")

        pl.seed_everything(42) # seed pytorch_lightning for reproducibility


    def load(self) -> None:

        # NOTE: order of models in dictionary defines models used for prediction at each index in load and solar arrays
        self.models = {
            model_type: {model_name: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,model_type,model_name)) for model_name in self.model_names[model_type]} for model_type in ['load','solar']
        }
        self.models.update({
            model_type: {self.model_names[model_type]: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,model_type,self.model_names[model_type]))} for model_type in ['pricing','carbon']
        })


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

    def reformat_df(self,df,ts_name,tv_cats):
        df = df.rename_axis(self.time_id_col_name).reset_index() # create column of indices to pass as time_idx to TimeSeriesDataSet - we have no missing values
        df[self.ts_id_col_name] = ts_name # create column with ID of timeseries (constant as only single timeseries)
        for col in tv_cats:
            df[col] = df[col].astype(str) # convert to strs to use as categoric covariates
        return df

    def format_CityLearn_datasets(
        self,
        CityLearn_dataset_dirpaths,
        model_type,
        building_index: int = None,
        ) -> List[TimeSeriesDataSet]:
        # format appropriate TimeSeriesDataSet from CityLearn dataset dir(s)
        # use construction from_dataset for datasets after first

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."

        if (model_type in ['load','solar']) and (building_index == None):
            raise ValueError(f"Must supply `building_index` to construct {model_type} dataset from CityLearn data.")

        building_fname_pattern = 'UCam_Building*.csv'

        # Specify column name variables.
        self.time_id_col_name = 'time_idx'
        self.ts_id_col_name = 'ts_id'
        self.load_col_name = 'Equipment Electric Power [kWh]'
        self.temp_col_name = 'Outdoor Drybulb Temperature [C]'
        self.solar_col_name = 'Solar Generation Power [kW]'
        self.dif_irad_col_name = 'Diffuse Solar Radiation [W/m2]'
        self.dir_irad_col_name = 'Direct Solar Radiation [W/m2]'
        self.pricing_col_name = 'Electricity Pricing [£]'
        self.carbon_col_name = 'Carbon Intensity [kg_CO2/kWh]'

        # Set default kwargs.
        if model_type == 'load':
            target = self.load_col_name
            ts_name = 'l'
            time_varying_unknown_reals = [self.load_col_name,self.temp_col_name]
        elif model_type == 'solar':
            target = self.solar_col_name
            ts_name = 's'
            time_varying_unknown_reals = [self.solar_col_name,self.dif_irad_col_name,self.dir_irad_col_name]
        elif model_type == 'pricing':
            target = self.pricing_col_name
            ts_name = 'p'
            time_varying_unknown_reals = [self.pricing_col_name]
        elif model_type == 'carbon':
            target = self.carbon_col_name
            ts_name = 'c'
            time_varying_unknown_reals = [self.carbon_col_name]

        self.time_varying_known_categoricals = ['Month','Hour','Day Type','Daylight Savings Status']

        ts_datasets = []

        for i,dirpath in enumerate(CityLearn_dataset_dirpaths):

            # Grab data from CityLearn files & environment.
            env = CityLearnEnv(os.path.join(dirpath,'schema.json'))
            # construct base dataframe with time info (known categoricals)
            first_building_file_path = glob.glob(os.path.join(dirpath,building_fname_pattern))[0]
            data_df = pd.read_csv(first_building_file_path,usecols=self.time_varying_known_categoricals)
            data_df = self.reformat_df(data_df, ts_name, self.time_varying_known_categoricals)

            # add type specific data to df, taking data from environment
            if model_type == 'load':
                data_df[self.load_col_name] = env.buildings[building_index].energy_simulation.non_shiftable_load
                data_df[self.temp_col_name] = env.buildings[building_index].weather.outdoor_dry_bulb_temperature
            elif model_type == 'solar':
                data_df[self.solar_col_name] = env.buildings[building_index].energy_simulation.solar_generation
                data_df[self.dif_irad_col_name] = env.buildings[building_index].weather.diffuse_solar_irradiance
                data_df[self.dir_irad_col_name] = env.buildings[building_index].weather.direct_solar_irradiance
            elif model_type == 'pricing':
                data_df[self.pricing_col_name] = env.buildings[0].pricing.electricity_pricing
            elif model_type == 'carbon':
                data_df[self.carbon_col_name] = env.buildings[0].carbon_intensity.carbon_intensity

            if i == 0:
                timeseries_dataset = TimeSeriesDataSet(
                    data_df,
                    time_idx=self.time_id_col_name,  # column name of time of observation
                    target=target,  # column name of target to predict
                    group_ids=[self.ts_id_col_name],  # column name(s) for timeseries IDs (static as only 1 timeseries used)
                    max_encoder_length=self.L,  # how much history to use
                    max_prediction_length=self.T,  # how far to predict into future
                    # covariates static for a timeseries ID - ignore for the moment
                    #static_categoricals=[ ... ],
                    #static_reals=[ ... ],
                    # covariates known and unknown in the future to inform prediction
                    time_varying_known_categoricals=self.time_varying_known_categoricals,
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
        assert train_dataset.max_encoder_length == self.L, f"`max_encoder_length` of input TimeSeriesDataSet {train_dataset.max_encoder_length} does not match encoder window of model group {self.L} (L)."
        assert train_dataset.max_prediction_length == self.T, f"`max_prediction_length` of input TimeSeriesDataSet {train_dataset.max_prediction_length} does not match planning horizon of model group {self.T} (T)."

        model_path = os.path.join(self.model_group_path,model_type,model_name)

        if os.path.exists(model_path):
            warnings.warn(f"Warning: A logs directory already exists for the model name `{model_name}`. By continuing you will overwrite this model.")
            if input("Are you sure you want to overwrite this model? [y/n]") not in ['y','yes','Y','Yes','YES']:
                print("Aborting model creation.")
                return
            else:
                shutil.rmtree(model_path)

        # Set default kwargs.
        # architecture hyperparameters
        if 'hidden_size' not in kwargs.keys(): kwargs['hidden_size'] = 48
        if 'attention_head_size' not in kwargs.keys(): kwargs['attention_head_size'] = 4
        if 'dropout' not in kwargs.keys(): kwargs['dropout'] = 0.1
        #if 'hidden_continuous_size' not in kwargs.keys(): kwargs['hidden_continuous_size'] = 16
        # loss metric to optimize
        if 'loss' not in kwargs.keys(): kwargs['loss'] = QuantileLoss()
        # set optimizer
        if 'optimizer' not in kwargs.keys(): kwargs['optimizer'] = 'adam'
        # optimizer parameters
        if 'reduce_on_plateau_patience' not in kwargs.keys(): kwargs['reduce_on_plateau_patience'] = 2

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


    def train_model(self, model_name: str, model_type: str, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, **kwargs) -> None:

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."
        assert model_name in self.model_names[model_type], f"Model {model_name} of type {model_type} not loaded into predictor."

        assert train_dataset.max_encoder_length == self.L, f"`max_encoder_length` of training dataset (TimeSeriesDataSet) {train_dataset.max_encoder_length} does not match encoder window of model group {self.L} (L)."
        assert train_dataset.max_prediction_length == self.T, f"`max_prediction_length` of training dataset (TimeSeriesDataSet) {train_dataset.max_prediction_length} does not match planning horizon of model group {self.T} (T)."
        assert val_dataset.max_encoder_length == self.L, f"`max_encoder_length` of validation dataset (TimeSeriesDataSet) {train_dataset.max_encoder_length} does not match encoder window of model group {self.L} (L)."
        assert val_dataset.max_prediction_length == self.T, f"`max_prediction_length` of validation dataset (TimeSeriesDataSet) {train_dataset.max_prediction_length} does not match planning horizon of model group {self.T} (T)."

        model_path = os.path.join(self.model_group_path,model_type,model_name)

        # continue training is checkpoint file available
        load_path_file = 'best_model.json'
        json_path = os.path.join(model_path,load_path_file)
        if os.path.exists(json_path):
            with open(json_path,'r') as json_file:
                best_model_chkpt = json.load(json_file)['rel_path']
            best_checkpoint_path = os.path.join(model_path,*best_model_chkpt)
        else:
            best_checkpoint_path = None

        # Get model.
        model = self.models[model_type][model_name]

        # Convert datasets to dataloaders for training.
        batch_size = 128
        n_workers = 4 if os.cpu_count() >= 8 else int(os.cpu_count()/2)
        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=n_workers)
        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=n_workers)

        # Create PyTorch Lightning Trainer with early stopping.
        # TODO: allow training kwargs to be taken in as method kwargs
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=5, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
        tb_logger = TensorBoardLogger(save_dir=model_path,name='')
        trainer = pl.Trainer(
            #deterministic=True, # remove stochasticity for reproducibility
            # set acceleration & hardware settings
            accelerator='auto',
            devices="auto",
            strategy="auto",
            # set training parameters
            max_epochs=100,
            enable_model_summary=False, # use True to see model structure & params
            gradient_clip_val=0.1,
            limit_train_batches=500,  # batches per epoch
            # set logging & callbacks
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


    def initialise_forecasting(self, num_buildings: int, tau: int):

        # define indices of variables in observations
        self.load_obs_index = 20
        self.solar_obs_index = 21
        self.pricing_obs_index = 24
        self.carbon_obs_index = 19

        # initialise observations buffer
        self.buffer = {
            'load': [[]*num_buildings],
            'solar': [[]*num_buildings],
            'pricing': [],
            'carbon': []
        }

        # check prediction models can provide specified tau
        assert self.T >= tau, f"Models cannot forecast for planning horizon {tau}, tau too large (> {self.T})."
        self.tau = tau


    def compute_forecast(self, observations, env: CityLearnEnv, t: int):

        assert all([len(self.models[key])>0 for key in self.models.keys()]), "You must load models for all variables to perform prediction."
        assert len(self.models['load']) == len(self.models['solar']) == np.array(observations).shape[0], "You must provide the same number of `load` and `solar` models as buildings being predicted for."

        # Update observation buffers.
        for model_type,obs_id in zip(self.model_types,[self.load_obs_index,self.solar_obs_index,self.pricing_obs_index,self.carbon_obs_index]):
            if model_type in ['load','solar']:
                for j,val in np.array(observations)[:, obs_id]:
                    self.buffer[model_type][j].append(val)
            else:
                self.buffer[model_type].append(np.array(observations)[0, obs_id])


        # Perform prediction.
        if (len(self.buffer['pricing']) < self.L) or (env.time_steps - t < self.T):
            return None # opt out of prediction if buffer not yet full
        else:
            # construct base df with time & past weather info
            months = env.buildings[0].energy_simulation.month[t-self.L+1:t+self.T+1]
            hours = env.buildings[0].energy_simulation.hour[t-self.L+1:t+self.T+1]
            day_types = env.buildings[0].energy_simulation.day_type[t-self.L+1:t+self.T+1]
            day_save_statuses = env.buildings[0].energy_simulation.daylight_savings_status[t-self.L+1:t+self.T+1]
            past_temps = env.buildings[0].weather.outdoor_dry_bulb_temperature[t-self.L+1:t+1]
            past_dif_irads = env.buildings[0].weather.diffuse_solar_irradiance[t-self.L+1:t+1]
            past_dir_irads = env.buildings[0].weather.direct_solar_irradiance[t-self.L+1:t+1]

            base_df = pd.DataFrame({
                'Month': months,
                'Hour': hours,
                'Day Type': day_types,
                'Daylight Savings Status': day_save_statuses,
                self.temp_col_name: np.append(past_temps,np.zeros(self.T)),
                self.dif_irad_col_name: np.append(past_dif_irads,np.zeros(self.T)),
                self.dir_irad_col_name: np.append(past_dir_irads,np.zeros(self.T))
            })
            base_df = self.reformat_df(base_df,'pred',self.time_varying_known_categoricals)

            # perform load forecasting
            load_forecasts = []
            for j,model in enumerate(self.models['load']):

                # construct data df 
                data_df = base_df.copy()
                data_df[self.load_col_name] = np.append(self.buffer['load'][j][-model.max_encoder_length:],np.zeros(model.max_prediction_length))

                # TODO: can I simply take in a dataframe for prediction or do I need to convert it to a dataloader or TimeSeriesDataSet?

                # perform prediction
                load_prediction = model.predict(data_df, mode='prediction')

                # save prediction in structure
                load_forecasts.append(load_prediction)

            if model_type == 'solar':
                target = self.solar_col_name
                ts_name = 's'
                time_varying_unknown_reals = [self.solar_col_name,self.dif_irad_col_name,self.dir_irad_col_name]
            elif model_type == 'pricing':
                target = self.pricing_col_name
                ts_name = 'p'
                time_varying_unknown_reals = [self.pricing_col_name]
            elif model_type == 'carbon':
                target = self.carbon_col_name
                ts_name = 'c'
                time_varying_unknown_reals = [self.carbon_col_name]

        return np.array(load_forecasts), ... # return structured predictions