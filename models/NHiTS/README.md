# Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)

Fill out ... and adjust lower text

## Package requirements

The top-level package required to use this model are provided in `requirements.txt`, which can be install using the following command,
```
pip install -r models/TFT/requirements.txt
```

Additionally [`numba`](https://numba.pydata.org/) can be `pip` installed to improve computation speeds.

If you would like to use CUDA to accelerate model training & evaluation, please run the following command *after* installing the base requirements,
```
pip3 install torch==1.13 --index-url https://download.pytorch.org/whl/cu117
```
see [this link](https://pytorch.org/get-started/locally/) for more details.

## Model organisation

The implemented wrapper class, `TFT_Predictor`, is a container for what is termed a 'model group'. A 'model group' consists of a collection of TFT model objects (instances of `pytorch_forecasting.TemporalFusionTransformer`) which are sufficient to perform inference for the predicted variables in a CityLearn environment/dataset, as well as a collection of metadata parameters. The 'model group' object acts as an interface between the CityLearn LinMPC object and the underlying TFT model objects.

The saved data for a trained model group is stored in a directory within `resources/model_logs` called `{model_group_name}`, which has sub-directories for each model type `('load','solar','pricing','carbon')`, each of which contain a sub-directory for each model of that type within the group, called `{model_name}`, containing log files and checkpoints for that model. The model group has a set of hyper-params common to all models it contains (encoder window length, `L`, and planning horizon, `T`), stored in the `model_group_params.json` file in the `{model_group_name}` directory. The model leaf directories contain a `best_model.json` which provides the path to the checkpoint file of the current best version of the model (used for loading convenience).

## Loading a predictor

Models can be loaded in a number of ways, however the recommended method is to load a complete model group with standard model naming during initialisation of the model group object via the following syntax,
```
UCam_ids = [...] # your building ids
TFT_group = TFT_Predictor(model_group_name, model_names=UCam_ids, load='group')
```
This loads models contained in the `model_logs/{model_group_name}` directory following the standard naming convention, corresponding to the ordered list of building ids passed as an argument. This method is preferred as it explicitly specifies the order of models loaded, and hence used during prediction.
The order of the models in the `TFT_Predictor.model_names` dict determines the order in which they are applied during prediction, and corresponds to the order of building objects in the `CityLearnEnv.buildings` list, i.e. `TFT_Predictor.model_names['load'][0]` is used to predict electrical load for building `CityLearnEnv.buildings[0]`.
**Be careful to check that the order of models used in your model group is as desired for prediction.**

For more info on loading models see the implementation of the `TFT_Predictor.__init__()` method.

## Using the predictor

Before prediction can be performed, the model must be set into prediction mode using the following command,
```
TFT_Predictor.initialise_forecasting(tau, env)
```
where `tau` is the desired forecasting horizon, and `env` is the `CityLearnEnv` object on which prediction is being done. This must be done *after* all models are loaded into the object.

Additionally, during prediction/inference the time step of the `CityLearnEnv` object must be pass as an extra argument so the appropriate time information can be gathered,

```
TFT_Predictor.compute_forecast(observations, env.time_step)
```

<br>

## Directory structure

The model directory contains the following:
- `TFT_predictor.py`: contains implementation of TFT model wrapper class
- `resources/model_logs`: directory containing save files for pre-trained models
- `create_train_TFT_group.py`: example script showing how to create a new TFT model group, train it on a specified dataset, and save it to `model_logs`
- `train_TFT_model.py`: example script showing how to load a model group and continue training for a specified model within it
- `infer_and_plot.py`: script performing inference using a specified model on specified dataset, and creating a dynamic timeseries plot of the results. Useful for checking the correct functioning of a trained model
- `requirements.txt`: contains top-level requirements to be `pip` installed

<br>

Example scripts provided should be run using the following syntax,
```
python3 -m models.TFT.create_train_TFT_group
```