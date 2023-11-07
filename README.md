# What's in this branch?

This branch contains the code used to perform the analyses comprising the Annex 37 sub-task A report. It consists of implementations of the prediction/forecasting methods investigated (found in the `models` directory), and scripts to assess and compare their performance (`assess_forecasts.py` & `evaluate.py`).

For information on the the setup of the task, see the `README` on the `main` branch.

# Directory structure

- `data`, datasets (from [Cambridge University Estates building electricity usage archive](https://github.com/EECi/Cambridge-Estates-Building-Energy-Archive)) used to perform simulations
- `models`, implementations of the prediction/forecasting methods investigated
&nbsp;Forecasting models implemented:
    - `dms`, simple Direct Multi-Step neural models
        - `Linear`, linear multi-layer perceptron (MLP) model
        - `Conv`, convolutional neural network (CNN) model
        - `ResMLP`, residual MLP model with skip-connections
    - `TFModels`, complex neural models implemented using [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/) framework
        - `TFT`, [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
        - `NHiTS`, [Neural Hierarchical Interpolaiton for Time Series Forecasting](https://arxiv.org/abs/2201.12886)
        - `DeepAR`, [DeepAR](https://arxiv.org/abs/1704.04110)
        - `RNN`, LSTM and GRU models
    - `noise`, explicit noising of perfect forecasts
    - `example`, example model implementation for template
- `experiments`, run scripts for performing experiments comprising analyses
- `results`, output files containing results of tests performed for conducted analyses
- `plots`, notebooks for plotting analysis results
- `utils`, supporting scripts
- `assess_forecasts.py`, script to test the quality/accuracy of forecasts provided by models
- `evaluate.py`, script to test the control performance of prediction models when used in the MPC
- `ground-truth.py`, script to test the control performance of the MPC if perfect forecasts were available
- `leaderboard.py`, script to update the leaderboard with the results of the above scripts
- `linmodel.py`, implementation of linear optimisation model used in MPC

<br>

# How to use the branch

Once you have a complete model implementation, you can add it to the library of methods in the `models` directory by doing the following:
- create a new sub-directory for your model - `models/<your model name>`
- wrap the implementation of your model in a class in its own script, e.g. `models/<your model name>/model.py` containing the class `MyModel`
    - this class must have a method `compute_forecast`, which takes in the array of current observations, and returns arrays for the forecasted variables - see documentation and example docstrings for required formatting
    - you should provide sensible default parameters (e.g. kwargs) so that a 'good' forecasting model is set up when an object of your class is constructed/initialised without arguments, e.g. load an up-to-date pre-trained model by default
- you should provide additional scripts as necessary to allow others to work with your model, e.g. for training, testing, interrogating, etc.
- put any required data files, e.g. pre-trained model specifications, in a sub-directory called `resources`
- update `models/__init__.py` to import your model class
- **provide a `README.md` in your model directory detailing: your model, the files you've provided, your preliminary results, any other important info**

An example model implementation directory is given at `models/example`.

# Model Comparison
We provide the following files for comparing the model's performance.

- `assess_forecast.py`
- `evaluate.py`
- `leaderboard.py`

Setting `save = True` in `assess_forecast.py` and `evaluate.py` will log the performances in the `outputs` directory.

Running `leaderboard.py` will then load the results from the `output` directory and update [leaderboard](archive_ignore/outputs/leaderboard.md) correspondingly.  

`assess_forecast.py` is used to assess the quality of the forecast only. 

`evaluate.py` uses the model's forecast for model predictive control, based on linear programming. This evaluates how good the forecast is for battery control.

If you wish to test your model in this framework you can run `assess_forecasts.py` and `evaluate.py`, editing the runtime sections of the scripts appropriately. *However it is recommended that you do this testing in your development branch before merging your model implementation*.

*Note: all scripts should be executed from the base `EECi` directory, and all file paths are specified relative to this.*