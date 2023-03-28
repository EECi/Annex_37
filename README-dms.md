# Direct Multi-Step Time Series Forecasting 🔮

---

## Files to Run 🏃‍♀️
`train.py`: contains the code for training the forecaster

`test.py`: contains the code for testing the forecaster

`test_individual.py`: contains the code for visualising the forecaster on a single dataset 

`inference.py`: contains the code for visualising the forecaster on a single dataset

---

# Train 🚂
Modify the parameters below and then the model by calling the train() method of the Predictor object.

###  L and T 🕰️  
These both variables that specify the length of the time series data that will be used to train the model:
- `L` determines the number of time steps in the input sequence. You can adjust this value to capture more or less 
  historical data.
- `T` determines the number of time steps in the output sequence. You can adjust this value to predict further into the 
  future.

### expt_name 🧪
is a string that specifies the name of the experiment you are running. A folder in the `logs` directory will
be generated with this name to track the experiment.

### mparam_dict 📜

mparam_dict is a dictionary that defines the model and the parameters to use for each dataset type. To use the same 
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

### model_name 🕵️‍♂️
Specifies the name of the model to load. Currently only the `vanilla` model has been implemented, which is an MLP. 

The architecture is defined using `L`, `T` and `layers` (a list that specifies the hidden layers dimensions).

### building_indices 🏘️
Specifies the buildings from the dataset to use (check the `/data` directory).

---

# Test
Modify the parameters below and then the model by calling the test() method of the Predictor object. This will evaluate 
all models inside the Predictor Object and append the results individually to a csv file. 

### expt_name 🧪
The `expt_name` for a trained model to load.

### results_file 👀
The CSV file for appending the results to. Will generate one if the specified name does not exist.

--- 

# Test Individual
Visualise the predictions of a trained model on an individual test dataset.

### expt_name 🧪
The `expt_name` for a trained model to load.

### building_index 🏘️
Index of the building to load the dataset from (check the `/data` directory).

### dataset_type ⌨️
A string to specify the dataset type to load ('solar', 'load', 'carbon', 'price').

--- 

# Inference
Visualise the inference of a trained model on an individual test dataset.

### expt_name 🧪
The `expt_name` for a trained model to load.

### building_index 🏘️
Index of the building to load the dataset from (check the `/data` directory).

### dataset_type ⌨️
A string to specify the dataset type to load ('solar', 'load', 'carbon', 'price').
