# Direct Multi-Step Time Series Forecasting 🔮


---

## Loading a pre-trained predictor

This is done in `evaluate.py` and `assess_forecasts.py`. The predictor will load the trained model 
specified by `expt_name` (see [`train.py`](#trainpy-)).

```python
from models import DMSPredictor
predictor = DMSPredictor(expt_name='linear_L168_T48', load=True)
```

## Running files inside the module 🏃‍♀️
The following files are provided inside the module, please run these from the root directory as a module, for example:
```
python -m models.dms.<filename>.py
```

More details on the each of the files can be found in later corresponding sections.

- [`train.py`](#trainpy-): contains the code for training the forecaster

- [`test.py`](#testpy-): contains the code for testing the forecaster

- [`test_individual.py`](#test_individualpy-): contains the code for visualising the forecaster on a single dataset 

- [`inference.py`](#inferencepy-): contains the code for visualising the forecaster on a single dataset

---

# train.py 🚂
Modify the parameters below and then train the model by calling the train() method of the Predictor object. The model 
will be trained until the maximum number of epochs is reached, or it will stop early if the validation performance does 
improve after a few epochs. The best model will automatically be saved.


You can use tensorboard during training by running the following in the command line. Replace ```<expt_name>``` with your 
corresponding experiment name or leave this empty.

```bash
tensorboard --logdir models/dms/resources/<expt_name>
```

###  `L` 🕰️  
Determines the number of time steps in the input sequence. You can adjust this value to capture more or less historical 
data.

###   `T` 🌅
Determines the number of time steps in the output sequence. You can adjust this value to predict further into the 
future.

### `expt_name` 🧪
A string that specifies the name of the experiment you are running. A folder in the `resources` directory will
be generated with this name to track the experiment.

### `mparam_dict` 📜

A dictionary that defines the model and the parameters to use for each dataset type. To use the same 
model and parameters for all dataset types use the following format:

```python
  mparam_dict = {'all': {'model_name': 'vanilla',
                         'mparam': {'L': L,
                                    'T': T,
                                    'layers': []}}}
```

Otherwise use the following format:
```python
  mparam_dict = {'solar': {'model_name': 'vanilla',
                           'mparam': {'L': L,
                                      'T': T,
                                      'layers': []}},
                 'load': {'model_name': 'vanilla',
                          'mparam': {'L': L,
                                     'T': T,
                                     'layers': []}},
                 'carbon': {'model_name': 'vanilla',
                            'mparam': {'L': L,
                                       'T': T,
                                       'layers': []}},
                 'price': {'model_name': 'vanilla',
                           'mparam': {'L': L,
                                      'T': T,
                                      'layers': []}}}
```

### `model_name` 🕵️‍♂️
Specifies the name of the model to load. Currently only the `vanilla` model has been implemented, which is an MLP. 

The architecture is defined using `L`, `T` and `layers` (a list that specifies the hidden layers dimensions).

### `building_indices` 🏘️
Specifies the buildings from the dataset to use (check the `/data` directory).

### `max_epoch` 💯
The maximum number of epochs to train the model if not stopped early.

### `patience` 🐢
The number of epochs the model will be trained with no improvement to the validation performance. After this number of 
epoch the training will be stopped early.


---

# test.py 🧫
Modify the parameters below and then test the model by calling the test() method of the Predictor object. This will 
evaluate all models inside the Predictor Class and append the results to a csv file or generate it if the specified
filename does not exist. 

### `expt_name` 🧪
The `expt_name` for a trained model to load.

### `results_file` 👀
The CSV file for appending the results to. Will generate one if the specified name does not exist.

--- 

# test_individual.py 🔬
Visualise the predictions of a trained model on an individual test dataset. The red line is the ground truth and the
blue lines are the predictions. Darker lines are predictions closer to the inference time and lighter lines are further 
predictions in the future.


### `expt_name` 🧪
The `expt_name` for a trained model to load.

### `building_index` 🏘️
Index of the building to load the dataset from (check the `/data` directory).

### `dataset_type` ⌨️
A string to specify the dataset type to load ('solar', 'load', 'carbon', 'price').

--- 

# inference.py 🤔
Visualise the inference of a trained model on an individual test dataset.

### `expt_name` 🧪
The `expt_name` for a trained model to load.

### `building_index` 🏘️
Index of the building to load the dataset from (check the `/data` directory).

### `dataset_type` ⌨️
A string to specify the dataset type to load ('solar', 'load', 'carbon', 'price').