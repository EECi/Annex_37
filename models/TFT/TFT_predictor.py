"""Implementation of TFT predictor model class."""

import os
import glob
import shutil
import json
import pickle
from typing import Any, List, Dict, Union

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from models.base_predictor_model import BasePredictorModel

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

# filer warnings: mostly object reload & numpy deprecation warnings
import warnings
warnings.filterwarnings(action='ignore',module=r'pytorch_forecasting')



class TFT_Predictor(BasePredictorModel):
    """Implementation of TFT-based prediction model for the CityLearn LinMPC controller.

    The provided class is used to perform the following activities:
    - load collections of pre-trained TFT models (all models required
        for prediction) - via `__init__` method.
    - format CityLearn datasets - via `format_CityLearn_datasets` method.
    - create new TFT models - via `new_model` method.
    - train TFT models on a provided dataset - via `train_model` method.
    - perform prediction inference - via `compute_forecast` method.

    A `TFT_Predictor` object contains/loads a 'model group', which is the
    collection of TFT modes required to produce all necessary predictions
    for the LinMPC controller.
    The parameters:
    - 'encoder window' (L)
    - 'planning horizon' (T)
    are set to be common for the model group and are saved as mparams
    for consistency in data handling for prediction.
    The models comprising the model group are saved into a 'model log'
    directory, with each type of model (load,solar,pricing,carbon)
    stored in a sub-directory of that name.

    NOTE: The order in which the models are stored in the `self.models` dict
    (which is the same as in `self.model_names`) specifies the order in which
    the models are used for prediction. So the index of the model must be the
    same as the index of the buildling you want to apply it to for prediction.
    EXTRA NOTE: When loading model groups implicitly be careful about the order
    in which models are loaded, as this may not be the way they display in your
    directory viewer. The ordering of strings is... complicated.
    """

    def __init__(self,
        model_group_name = 'default',
        L: int = 72,
        T: int = 48,
        model_names: Union[List, Dict] = None,
        load: Union[str, bool] = 'group'
        ) -> None:
        """Create model group object, loading models from file or initialising
        empty model group to be populated later.

        Args:
            model_group_name (str, optional): Name of model group. Defaults to 'default'.
            Gives name of model log diretory for model group.
            L (int, optional): Model group encoder window parameter. Defaults to 72.
            T (int, optional): Model group planning horizon parameter. Defaults to 48.
            model_names (Union[List, Dict], optional): Dictionary of names of models of
            each type to load, or list of indices to quickly specify models named `load_{index}`,
            `solar_{index}`, `pricing`, `carbon`. Defaults to None. Not required is loading a
            model group.
            load (Union[str, bool], optional): Whether to load a model group ('group'),
            load a custom selection of models ('indiv') as specified in the `model_names`
            dict, or not load any models and initialise a blank model group. Defaults to 'group'.
        """

        # Specify model parameters.
        self.model_types = ['load','solar','pricing','carbon']
        self.model_group_name = model_group_name
        self.model_group_path = os.path.join('models','TFT','resources','model_logs',model_group_name)

        # Specify column name variables for data formatting.
        self.time_id_col_name = 'time_idx'
        self.ts_id_col_name = 'ts_id'
        self.load_col_name = 'Equipment Electric Power [kWh]'
        self.temp_col_name = 'Outdoor Drybulb Temperature [C]'
        self.solar_col_name = 'Solar Generation Power [kW]'
        self.dif_irad_col_name = 'Diffuse Solar Radiation [W/m2]'
        self.dir_irad_col_name = 'Direct Solar Radiation [W/m2]'
        self.pricing_col_name = 'Electricity Pricing [£]'
        self.carbon_col_name = 'Carbon Intensity [kg_CO2/kWh]'

        # Specify common categoric covariates for models.
        self.time_varying_known_categoricals = ['Month','Hour','Day Type','Daylight Savings Status']


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
            assert [key in list(model_names.keys()) for key in self.model_types]
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
                    assert os.path.exists(os.path.join(self.model_group_path,model_type)), f"`{model_type}` model logs sub-dir does not exist."
                for model_type in ['pricing','carbon']:
                    if len(list(os.scandir(os.path.join(self.model_group_path,model_type)))) > 1:
                        warnings.warn("Warning: More than 1 {} model available. 1st selected in group load.".format(model_type))
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
                warnings.warn("Warning: A log directory for this model group name ({}) already exists but you are not loading it.".format(model_group_name))
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
        """Load models specified in `self.model_names` into `self.models`.
        Model objects are instances of `pytorch_lightning.TemporalFusionTransformer`.
        """

        # NOTE: order of models in dictionary defines models used for prediction at each index in load and solar arrays
        self.models = {
            model_type: {model_name: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,model_type,model_name)) for model_name in self.model_names[model_type]} for model_type in ['load','solar']
        }
        self.models.update({
            model_type: {self.model_names[model_type]: self.load_TFT_from_model_dir_path(os.path.join(self.model_group_path,model_type,self.model_names[model_type]))} for model_type in ['pricing','carbon']
        })


    def load_TFT_from_model_dir_path(self, model_path) -> TemporalFusionTransformer:
        """Load best version of saved model using path to directory.

        NOTE: `best_model.json` contains a dict with one key `rel_path`, whose values is a list
        of the relative path (strings) to the checkpoint file of the best version of that model.
        This file is used to conventiently track the best version of the model trained so far
        for ease of loading and continuation of training.

        Args:
            model_path (Union[str, os.Path]): path to model log directory of desired model.

        Returns:
            TemporalFusionTransformer: Loaded model object (loaded from best checkpoint file).
        """

        assert os.path.exists(model_path), f"The requested model `{model_path}` does not exist."

        load_path_file = 'best_model.json'
        json_path = os.path.join(model_path,load_path_file)
        assert os.path.exists(json_path), f"JSON indicating checkpoint file to load `{json_path}` does not exist."

        with open(json_path,'r') as json_file:
            best_model_chkpt = json.load(json_file)['rel_path']
        best_model_path = os.path.join(model_path, *best_model_chkpt)

        return TemporalFusionTransformer.load_from_checkpoint(best_model_path)



    def reformat_df(self,df,ts_name,tv_cats):
        """Reformat CityLearn dataset `pd.DataFrame` for construcction into
        `pytorch_lightning.TimeSeriesDataSet`.
        Converts categoricals to strs and adds timeseries id column.

        Args:
            df (pd.DataFrame): DataFrame to reformat.
            ts_name (str): Timeseries id label to add.
            tv_cats (List[str]): Column names of categorical variables to re-type.

        Returns:
            df (pd.DataFrame): Reformatted DataFrame
        """
        df = df.rename_axis(self.time_id_col_name).reset_index() # create column of indices to pass as time_idx to TimeSeriesDataSet - we have no missing values
        df[self.ts_id_col_name] = ts_name # create column with ID of timeseries (constant as only single timeseries)
        for col in tv_cats:
            df[col] = df[col].astype(str) # convert to strs to use as categoric covariates
        return df


    def make_TimeSeriesDataSet(self, data_df: pd.DataFrame, target: str, time_varying_unknown_reals: List[str]):
        """Format pandas.DataFrame dataset into pytorch_forecasting.TimeSeriesDataSet
        using group model attributes.

        Args:
            data_df (pd.DataFrame): Dataset to be formatted.
            target (str): Column name of target variable in input dataset.
            time_varying_unknown_reals (List[str]): List of time varying unknown real
            covariates for desired target variable.

        Returns:
            timeseries_dataset (TimeSeriesDataSet):Formatted dataset.
        """

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

        return timeseries_dataset


    def get_TimeSeriesDataSet_parameters(self, model_type: str, model_name: str) -> Dict:
        """Load TimeSeriesDataSet parameters for trained model from json."""

        dataset_params_path = os.path.join(self.model_group_path,model_type,model_name,'timeseries_dataset_params.pkl')
        with open(dataset_params_path,'rb') as pkl_file:
            tsds_params = pickle.load(pkl_file)

        return tsds_params


    def format_CityLearn_datasets(
        self,
        CityLearn_dataset_dirpaths,
        model_type,
        model_name: str = None,
        building_index: int = None,
        ) -> List[TimeSeriesDataSet]:
        """Format CityLearn datasets into `pytorch_lightning.TimeSeriesDataSet`
        objects for use with pytorch_lightning objects, e.g. for training.
        The constructed dataset is for a specified prediction variable, given
        by the `model_type` and `building_index` (index of building in
        CityLearnEnv.buildings list) for building specified variables (load
        and solar).

        NOTE:
        - CityLearn datasets are specified by the path to their directory.
        - multiple datasets can be passed for formatting. If so they are
        all formatted in the same way, with the first used as a template
        for the rest.

        Args:
            CityLearn_dataset_dirpaths (List[Union[str, os.Path]]): List of
            paths to CityLearn datasets to be formatted.
            model_type (str): Type of prediction variable to construct dataset
            for, one of ['load','solar','pricing','carbon'].
            model_name (str): Name of model to construct dataset for. Used when the
            model already exists. Timeseries dataset parameters for encoding and
            scaling the input data are collected from file to ensure dataset
            consistency. ALWAYS USE unless creating a new model.
            building_index (int, optional): Index of building in CityLearnEnv.buildings
            of target variable for dataset. Defaults to None. Required if target
            variable is building specified, i.e. `model_type` is 'load' or 'solar'.

        Returns:
            List[TimeSeriesDataSet]: List of formatted datasets created.
        """
        # format appropriate TimeSeriesDataSet from CityLearn dataset dir(s)
        # use construction from_dataset for datasets after first

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."

        if (model_type in ['load','solar']) and (building_index == None):
            raise ValueError(f"Must supply `building_index` to construct {model_type} dataset from CityLearn data.")

        if model_name is not None:
            assert model_name in self.model_names[model_type], f"Model {model_name} of type {model_type} not loaded into predictor."
            tsds_params = self.get_TimeSeriesDataSet_parameters(model_type,model_name)
        else:
            warnings.warn("Warning: Creating new TimeSeriesDataSet object with new encodings/scalings. Note, only to be used for training a new model.")

        building_fname_pattern = 'UCam_Building*.csv'

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

            if model_name is not None: # make dataset from params of original train set
                timeseries_dataset = TimeSeriesDataSet.from_parameters(tsds_params, data_df)
            else:
                if i == 0: # create new TimeSeriesDataSet (and encoding/scaling)
                    timeseries_dataset = self.make_TimeSeriesDataSet(data_df,target,time_varying_unknown_reals)
                else: # make dataset from params of first data set
                    timeseries_dataset = TimeSeriesDataSet.from_dataset(ts_datasets[0], data_df)

            ts_datasets.append(timeseries_dataset)

        return ts_datasets



    def new_model(self, model_name: str, model_type: str, train_dataset: TimeSeriesDataSet, pre_confirm=False, **kwargs) -> TemporalFusionTransformer:
        """Create a new TFT model object with format given by provided TimeSeriesDataSet.

        Args:
            model_name (str): Name of model. Sets name of model log directory.
            model_type (str): Type of variable predicted by model, one of
            ['load','solar','pricing','carbon'].
            train_dataset (TimeSeriesDataSet): Dataset object used to format model.
            pre_confirm (bool): Whether to skip overwrite warning step (e.g. during
            programatic model creation).
        
        NOTE: If the specified model name already exists within the model group,
        the data for the existing model will be overwritten when the new model is
        created.

        Returns:
            TemporalFusionTransformer: Created model.
        """

        assert model_type in self.model_types, f"`model_type` argument must be one of {self.model_types}."
        assert type(train_dataset) == TimeSeriesDataSet, "`train_dataset` must be a pytorch_forecasting.TimeSeriesDataSet object."
        assert train_dataset.max_encoder_length == self.L, f"`max_encoder_length` of input TimeSeriesDataSet {train_dataset.max_encoder_length} does not match encoder window of model group {self.L} (L)."
        assert train_dataset.max_prediction_length == self.T, f"`max_prediction_length` of input TimeSeriesDataSet {train_dataset.max_prediction_length} does not match planning horizon of model group {self.T} (T)."

        model_path = os.path.join(self.model_group_path,model_type,model_name)

        if os.path.exists(model_path):
            warnings.warn("Warning: A logs directory already exists for the model name `{}`. By continuing you will overwrite this model.".format(model_name))
            if pre_confirm:
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
        if 'reduce_on_plateau_patience' not in kwargs.keys(): kwargs['reduce_on_plateau_patience'] = 3

        # initialise TFT model from train_dataset specification
        tft = TemporalFusionTransformer.from_dataset(train_dataset,**kwargs)

        # set model
        if model_type in ['load','solar']:
            self.model_names[model_type].append(model_name)
        else:
            warnings.warn("Warning: f{} model replaced.".format(model_type))
            self.model_names[model_type] = model_name
        self.models[model_type][model_name] = tft

        os.makedirs(os.path.realpath(model_path))

        # save timeseries dataset parameters in model dir
        # NOTE!!!: this defines the variable scaling and encoding and so must
        # be used to construct all following datasets applied to the model
        tsds_params = train_dataset.get_parameters()
        dataset_params_path = os.path.join(model_path,'timeseries_dataset_params.pkl')
        with open(dataset_params_path,'wb') as pkl_file:
            pickle.dump(tsds_params, pkl_file)

        return tft


    def train_model(self, model_name: str, model_type: str, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, **kwargs) -> None:
        """Train specified model. Training will continue from the new available model
        if a saved version already exists.

        Args:
            model_name (str): Name of model to train.
            model_type (str): Type of model to train.
            train_dataset (TimeSeriesDataSet): Training dataset to use for training.
            val_dataset (TimeSeriesDataSet): Validation dataset to use for training.
        """

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
            max_epochs=50,
            enable_model_summary=False, # use True to see model structure & params
            gradient_clip_val=0.1,
            limit_train_batches=800,  # batches per epoch
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
            if current_level != model_name: path_components.insert(0, current_level)
        best_model_path_dict = {'rel_path':path_components}

        with open(json_path,'w') as json_file:
            json.dump(best_model_path_dict, json_file)



    def initialise_forecasting(self, tau: int, env: CityLearnEnv):
        """Initialise attributes required to perform forecasting.

        NOTE: This must be done after object initialisation as
        additional models may be added to the group, and the
        number of models used for prediction are not known until
        all (potentially custom) loading steps are complete.

        Args:
            tau (int): Forecasting horizon to used during prediction.
            Must be less than the planning horizon of the model group.
            env (CityLearnEnv): CityLearnEnvironment object.
        """

        # check prediction models can provide specified tau
        assert self.T >= tau, f"Models cannot forecast for planning horizon {tau}, tau too large (> {self.T})."
        self.tau = tau

        # define indices of variables in observations
        self.load_obs_index = 20
        self.solar_obs_index = 21
        self.pricing_obs_index = 24
        self.carbon_obs_index = 19

        # initialise observations buffer
        self.buffer = {
            'load': [[] for l in range(len(self.models['load'].keys()))],
            'solar': [[] for l in range(len(self.models['solar'].keys()))],
            'pricing': [],
            'carbon': []
        }

        # grab time data from environment & store
        self.simulation_duration = env.time_steps
        self.months = env.buildings[0].energy_simulation.month
        self.hours = env.buildings[0].energy_simulation.hour
        self.day_types = env.buildings[0].energy_simulation.day_type
        self.day_save_statuses = env.buildings[0].energy_simulation.daylight_savings_status
        self.past_temps = env.buildings[0].weather.outdoor_dry_bulb_temperature
        self.past_dif_irads = env.buildings[0].weather.diffuse_solar_irradiance
        self.past_dir_irads = env.buildings[0].weather.direct_solar_irradiance

        # put models in evaluation mode for prediction
        for model_type in self.models.keys():
            for model in self.models[model_type].values():
                model.eval()


    def compute_forecast(self, observations, t: int):
        """Perform prediction inference required for CityLearn LinMPC controller.

        NOTE: The CityLearnEnv object for which predictions are being made
        is taken an argument here solely for the convenience of getting the
        time info from the CityLearn environment rather than keeping track
        of datetime objects.
        Yes you have access to all the true data. JUST DON'T CHEAT!

        Args:
            observations (List[List]): Observations array as specified
            by CityLearn environment
            t: current value of `env.time_step` (i.e. when prediction
            is made)

        NOTE: At the time of prediction, `env.time_step` (t) is the index of
        the observations we have just received in the internal data lists
        of the CityLearnEnv object. So the encoder window indices are [t-L+1,t]
        (inclusive), and the planning horizon indices are [t+1,t+T] (inclusive)

        Returns:
            predicted_loads (np.array): predicted electrical loads of buildings in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pv_gens (np.array): predicted energy generations of pv panels in each
                period of the planning horizon (kWh) - shape (N,tau)
            predicted_pricing (np.array): predicted grid electricity price in each period
                of the planning horizon ($/kWh) - shape (tau)
            predicted_carbon (np.array): predicted grid electricity carbon intensity in each
                period of the planning horizon (kgCO2/kWh) - shape (tau)
        """

        assert hasattr(self, 'buffer'), "You must enter prediction mode by calling `self.initialise_forecasting(tau)` before forecasting can be performed."
        assert all([len(self.models[key])>0 for key in self.models.keys()]), "You must load models for all variables to perform prediction."
        assert len(self.models['load']) == len(self.models['solar']) == np.array(observations).shape[0], "You must provide the same number of `load` and `solar` models as buildings being predicted for."

        # Update observation buffers.
        for model_type,obs_id in zip(self.model_types,[self.load_obs_index,self.solar_obs_index,self.pricing_obs_index,self.carbon_obs_index]):
            if model_type in ['load','solar']:
                for j,val in enumerate(np.array(observations)[:, obs_id]):
                    self.buffer[model_type][j].append(val)
            else: # pricing and carbon observations are shared between buildings
                self.buffer[model_type].append(np.array(observations)[0, obs_id])


        # Perform prediction.
        if (len(self.buffer['pricing']) < self.L) or (self.simulation_duration - t < self.T):
            return None # opt out of prediction if buffer not yet full
        else:
            # construct base df with time & past weather info
            months = self.months[t-self.L+1:t+self.T+1]
            hours = self.hours[t-self.L+1:t+self.T+1]
            day_types = self.day_types[t-self.L+1:t+self.T+1]
            day_save_statuses = self.day_save_statuses[t-self.L+1:t+self.T+1]
            past_temps = self.past_temps[t-self.L+1:t+1]
            past_dif_irads = self.past_dif_irads[t-self.L+1:t+1]
            past_dir_irads =self.past_dir_irads[t-self.L+1:t+1]

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
            predicted_loads = []
            for j,model_name in enumerate(self.model_names['load'].values()):
                model = self.models['load'][model_name]
                tsds_params = self.get_TimeSeriesDataSet_parameters(model_type,model_name)
                data_df = base_df.copy()
                data_df[self.load_col_name] = np.append(self.buffer['load'][j][-self.L:],np.zeros(self.T))
                data_ds = TimeSeriesDataSet.from_parameters(tsds_params, data_df)
                load_prediction = np.array(model.predict(data_ds, mode='prediction')).reshape(self.T)[:self.tau]
                predicted_loads.append(load_prediction)
            predicted_loads = np.array(predicted_loads)

            # perform solar forecasting
            predicted_pv_gens = []
            for j,model_name in enumerate(self.model_names['solar'].values()):
                model = self.models['load'][model_name]
                tsds_params = self.get_TimeSeriesDataSet_parameters(model_type,model_name)
                data_df = base_df.copy()
                data_df[self.solar_col_name] = np.append(self.buffer['solar'][j][-self.L:],np.zeros(self.T))
                data_ds = TimeSeriesDataSet.from_parameters(tsds_params, data_df)
                solar_prediction = np.array(model.predict(data_ds, mode='prediction')).reshape(self.T)[:self.tau]
                predicted_pv_gens.append(solar_prediction)
            predicted_pv_gens = np.array(predicted_pv_gens)

            # perform pricing forecasting
            model_name = list(self.model_names['pricing'].values())[0]
            model = self.models['pricing'][model_name]
            tsds_params = self.get_TimeSeriesDataSet_parameters(model_type,model_name)
            data_df = base_df.copy()
            data_df[self.pricing_col_name] = np.append(self.buffer['pricing'][-self.L:],np.zeros(self.T))
            data_ds = TimeSeriesDataSet.from_parameters(tsds_params, data_df)
            predicted_pricing = np.array(model.predict(data_ds, mode='prediction')).reshape(self.T)[:self.tau]

            # perform carbon forecasting
            model_name = list(self.model_names['carbon'].values())[0]
            model = self.models['carbon'][model_name]
            tsds_params = self.get_TimeSeriesDataSet_parameters(model_type,model_name)
            data_df = base_df.copy()
            data_df[self.carbon_col_name] = np.append(self.buffer['carbon'][-self.L:],np.zeros(self.T))
            data_ds = TimeSeriesDataSet.from_parameters(tsds_params, data_df)
            predicted_carbon = np.array(model.predict(data_ds, mode='prediction')).reshape(self.T)[:self.tau]

        return predicted_loads, predicted_pv_gens, predicted_pricing, predicted_carbon