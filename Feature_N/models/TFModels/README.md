# `pytorch-forecasting` Models

This directory provides an interface implementation for the [PyTorch Forecasting library](https://pytorch-forecasting.readthedocs.io/en/stable/index.html) and its models for use in the CityLearn LinMPC task. A base interface wrapper class is implemented, with child classes used for each model architecture available in the PyTorch Forecasting framework - [`DeepAR`, `NHiTS`, `TemporalFusionTransformer`, `RNN`].

## Package requirements

The top-level package required to use this model are provided in `requirements.txt`, which can be install using the following command,

```
pip install -r models/TFModels/requirements.txt
```

Additionally [`numba`](https://numba.pydata.org/) can be `pip` installed to improve computation speeds.

If you would like to use CUDA to accelerate model training & evaluation, please run the following command *after* installing the base requirements,

```
pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

see [this link](https://pytorch.org/get-started/locally/) for more details.

## Model organisation

The implemented base wrapper class, `TF_Predictor`, is a container for what is termed a 'model group'. Each model wrapper class inherits from `TF_Predictor` and implements the features specific to its `model_architecture`. A 'model group' consists of a collection of model objects (instances of `pytorch_forecasting.{model_architecture}`) which are sufficient to perform inference for the predicted variables in a CityLearn environment/dataset, as well as a collection of metadata parameters. The 'model group' object acts as an interface between the CityLearn LinMPC object and the underlying model objects.

The saved data for a trained model group is stored in a directory within `{model_architecture}/resources/model_logs` called `{model_group_name}`, which has sub-directories for each model type `('load','solar','pricing','carbon')`, each of which contain a sub-directory for each model of that type within the group, called `{model_name}`, containing log files and checkpoints for that model. The model group has a set of hyper-params common to all models it contains (encoder window length, `L`, and planning horizon, `T`), stored in the `model_group_params.json` file in the `{model_group_name}` directory. The model leaf directories contain a `best_model.json` which provides the path to the checkpoint file of the current best version of the model (used for loading convenience).

## Loading a predictor

Models can be loaded in a number of ways, however the recommended method is to load a complete model group with standard model naming during initialisation of the model group object via the following syntax,

```
from models import {model_architecture}_Predictor as Predictor
UCam_ids = [...] # your building ids
model_group = Predictor(model_group_name, model_names=UCam_ids, load='group')
```

This loads models contained in the `model_logs/{model_group_name}` directory following the standard naming convention, corresponding to the ordered list of building ids passed as an argument. This method is preferred as it explicitly specifies the order of models loaded, and hence used during prediction.
The order of the models in the `Predictor.model_names` dict determines the order in which they are applied during prediction, and corresponds to the order of building objects in the `CityLearnEnv.buildings` list, i.e. `Predictor.model_names['load'][0]` is used to predict electrical load for building `CityLearnEnv.buildings[0]`.
**Be careful to check that the order of models used in your model group is as desired for prediction.**

For more info on loading models see the implementation of the `Predictor.__init__()` method.

## Using the predictor

Before prediction can be performed, the model must be set into prediction mode using the following command,

```
Predictor.initialise_forecasting(tau, env)
```

where `tau` is the desired forecasting horizon, and `env` is the `CityLearnEnv` object on which prediction is being done. This must be done *after* all models are loaded into the object.

Additionally, during prediction/inference the time step of the `CityLearnEnv` object must be pass as an extra argument so the appropriate time information can be gathered,

```
Predictor.compute_forecast(observations, env.time_step)
```

<br>

## Directory structure

The model directory contains the following:

- `TFPredictor.py`: implementation of base wrapper class for interfacing `pytorch_forecasting` models with CityLearn environment for inference, as well as common activities such as loading models, generating compatible datasets, training, etc.
- `{model_architecture}` directories: containing model implementations for each `model_architecture`. Each contains:
    - `README.md`: explanation of the model architecture
    - `{model_architecture}_predictor.py`: containing implementation of model wrapper class for that architecture
    - `resources/model_logs`: directory containing save files for pre-trained models
- `create_train_TF_group.py`: example script showing how to create a new model group, train it on a specified dataset, and save it to `model_logs`
- `train_TF_model.py`: example script showing how to load a model group and continue training for a specified model within it
- `infer_and_plot.py`: script performing inference using a specified model on specified dataset, and creating a dynamic timeseries plot of the results. Useful for checking the correct functioning of a trained model
- `requirements.txt`: contains top-level requirements to be `pip` installed

<br>

Example scripts provided should be run using the following syntax,

```
python3 -m models.TFModels.<script_name>
```
