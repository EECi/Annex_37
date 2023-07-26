# What's in this branch?

This branch collates the outputs of the contributions to the Annex 37 Sub-task A work packet, and is used to perform the combined analyses presented in the report. It contains implementations of the prediction/forecasting methods investigated (found in the `models` directory), and scripts to assess and compare their performance (`assess_forecasts.py` & `evaluate.py`).

For information on the the setup of the task, see the `README` on the `main` branch.

<br>

# EECi's approach

The following table outlines the research questions that the members of [EECi](https://eeci.github.io/home/) are investigating as part of the Annex 37 task, and the prediction methods that they are working with to do so.

| Name                                  | Research Question                                                                                                                                                                                                                                                                                                                                                                                                         | Model/Method                                                                                                                                                                              | Directory                            | Progress                                                                                                       |
| :------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Example                               | Can we build the infrastructure?                                                                                                                                                                                                                                                                                                                                                                                          | Linear Interpolation                                                                                                                                                                      | [`example`](models/example/README.md) | Complete ✅                                                                                                    |
| [Max](mailto:mal84@cam.ac.uk)            | - How does the volume of available of training data, and the similarity of training data to test data, affect prediciton performance for neural methods?<br> - How much data do we need for training, and can we use data from other buildings?                                                                                                                                                                      | - [Temporal Fusion Transformers (TFTs)](https://arxiv.org/abs/1912.09363) <br> - [Neural Hierarchical Interpolation for Time Series Forecasting (N-HiTS)](https://arxiv.org/abs/2201.12886) <br> - [DeepAR](https://arxiv.org/abs/1704.04110) | `TFModels`                                  | In testing ⚙️                                                                                                |
| [Pat](mailto:vw273@cam.ac.uk)            | - Which methods are best for time-series forecasting?<br> - Why is this the case (inductive biases)?                                                                                                                                                                                                                                                                                                                  | - Direct multi-step prediction <br> - Confidence score prediction <br> - Ensembling <br> - Action-supervised forecasting                                                      | [`dms`](models/dms/README.md)         | - Direct Multi-step forecasting with MLP implemented ✅<br> - About to start confidence score stuff 👨‍🔧 |
| [Nick](mailto:nm735@cam.ac.uk)           | - Can we create generalised building-type training data sets to use for “typical buildings” in absence of building-specific training data?<br> - How do we fine-tune a pre-trained predictor? <br> How building-specific does the training set need to be? <br> - Comparison of predictors trained with specific building data vs generalised building data                                                 | - Any supervised model, especially easy/fast to train <br> - Statistical comparison methods & metrics                                                                                 | tbc                                  |                                                                                                                |
| [Monika](mailto:mk2040@cam.ac.uk)        | - How do we classify the behaviours/trends within the time-series?<br> - How do we detect change in behaviours/trends in the time-series?<br> - How does model selection change upon changes in trend?                                                                                                                                                                                                            | Online change point detection algorithms, e.g. Bayesian Online Changepoint Detection (leading to Bayesian On-line Changepoint Detection with Model Selection (BOCPDMS) if time permits)   | tbc                                  |                                                                                                                |
| [Rebecca](mailto:rward@turing.ac.uk)     | - What are the similarities and differences between the demand profiles for different buildings?                                                                                                                                                                                                                                                                                                                          | FDA                                                                                                                                                                                       | tbc                                  |                                                                                                                |
| [Zack](mailto:zxuerebconti@turing.ac.uk) | -Can we infer representative dynamics from observed data to build more robust and interpretable predictors? How does this influence training volume efficiency?<br> - Can we leverage coarse Physics-based models to improve training efficiency and predictor robustness? <br> -  Can we obtain simplified models with the least number of tunable hyperparameters possible as required in control applications? | - Dynamic Mode Decomposition with control (DMDc)<br> - High-order Dynamic Mode Decomposition (HODMD) <br> -  LSTM/PINN with RC models                                             | tbc                                  |                                                                                                                |
| [Chaoqun](mailto:czhuang@turing.ac.uk)   | -  How do we combine offline and online learning methods for time-series forecasting? Would the FLOPs meet the limits of control devices？<br> - How to determine the optimal update interval for online learning?  <br> - The impacts of exogenous variables and how to select them.                                                                                                                             | LSTM, GRU, Spearman Correlation                                                                                                                                                           | tbc                                  |                                                                                                                |

## Get in touch

If you would like to find out more about the approaches we are taking, would like to get involved with the work items, or have any suggestions for research questions or methods, please either email us using the links above or fill in [this form](https://forms.gle/KCmPPjirVn6TRkQJ6).

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

Running `leaderboard.py` will then load the results from the `output` directory and update
[leaderboard](archive_ignore/outputs/leaderboard.md) correspondingly.  

`assess_forecast.py` is used to assess the quality of the forecast only. 

`evaluate.py` uses the model's forecast for model predictive control, based on linear programming. This evaluates how 
good the forecast is for battery control.

If you wish to test your model in this framework you can run `assess_forecasts.py` and `evaluate.py`, editing the runtime sections of the scripts appropriately. *However it is recommended that you do this testing in your development branch before merging your model implementation*.

*Note: all scripts should be executed from the base `EECi` directory, and all file paths are specified relative to this.*