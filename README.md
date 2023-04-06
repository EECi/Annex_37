
# Annex 37

## Smart design and control of energy storage systems 

The aim of [Annex 37](https://iea-es.org/task-37/) is to investigate smart design and control strategies for energy storage systems at both the supply side and demand side. As part of Sub-task A - Demand and Supply Prediction - we aim to investigate the role of data in enabling optimised battery scheduling in smart buildings. This repository contains datasets and a framework for evaluating the role of data in optimising battery scheduling in building-based micro-grids with distributed generation & storage.


## Proposed Task

As part of the Sub-task A work package it is proposed to investigate the role of data in enabling optimised battery scheduling in smart buildings with distributed generation (solar PV) & battery storage.

This investigation will seek to analyse the impact of different aspects of data usage on the development of prediction/forecasting methods for use within Linear Model Predictive Control (MPC) algorithms for optimising battery scheduling in micro-grids. This is done to provide a common framework within which the impact of data on the performance of building energy system control can be analysed. The analysis will be conducted using the [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) model of electrical micro-grids composed of building loads, considering only the direct electricity usage of buildings (i.e. not accounting for heating related electrical demand). A novel dataset comprised of electrical demand data from the Cambridge University Estates, and open weather, carbon, and pricing data will be used for the analysis.

The investigation seeks to answer questions such as the following:
- How much data is needed to achieve performant battery scheduling systems?
- What is the cost-performance trade-off when acquiring data to train forecasting models for building control?
- Which variables are most important to forecast accurately to achieve good performance? Or, where should forecasting effort be expended and where can it be economised?
- When and where should measurements be made to best train forecasting models?
- How do different methods compare in terms of data efficiency, and how can the most be made of available data?

Participants are encouraged to explore different forecasting methods and different aspects of the impact of data on the optimal scheduling scenario.


## The CityLearn Environment

[CityLearn](https://github.com/intelligent-environments-lab/CityLearn) is a framework for studying the performance of control algorithms for optimising battery scheduling in micro-grids of building with both distributed solver PV generation & battery storage, with a particular focus on Reinforcement Learning (RL) based methods.

The CityLearn environment is used in annual challenges hosted by the [Intelligent Environment Lab](https://www.ie-lab.org/) to explore different aspects of RL algorithm design and behaviour. In the [CityLearn 2022 challenge](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge), run as part of [NeurIPS 2022](https://nips.cc), international teams competed to minimise costs and carbon emissions across a group of residential buildings equipped with solar PV and battery storage. The teams implemented a variety of control strategies, including: rule-based controllers, model predictive controllers or reinforcement learning with single agent or multi agent set-up.

In the first phase of the competition, demand data for 5 buildings were provided and teams were able to train their models on the supplied data.  In phase 2, the models were applied to an unseen dataset for another 5 buildings, and in phase 3 to an unseen dataset for a final 7 buildings.  The final ranking was calculated based on the performance over all 3 datasets.

Final performance was judged on three criteria:

1)  minimum normalised cost of electricity taken from the grid supply
2)  minimum normalised carbon emissions
3)  minimum [grid cost](https://discourse.aicrowd.com/t/announcement-update-to-evaluation-and-leaderboard/8156)

The objective contributions were all normalised with respect to the values with no operational battery, and the overall objective was an evenly weighted sum of the three contributions.

From this challenge, it was identified that Model Predictive Control (MPC) was an attractive method for battery scheduling, providing good performance within a clear and explainable framework that shows promise for generalisation and the incorporation of prior information. The proposal for Sub-task A seeks to explore this method further.


## Explanation of Linear MPC

For a brief overview of how Linear Model Predictive Control works, and how it is implemented in the context of the CityLearn environment, please see [this tutorial](https://colab.research.google.com/drive/1Qzs4GhL-OZCY3YoFsVSzA8_BpTnL518P?usp=sharing).


## Available Data

For the analysis, hourly-resolved building electrical load from the Cambridge University Estates portfolio will be used. Metering data from 30 buildings of significantly varying behaviours (temporal trends & scales) for the 10 year period 2010-2019 (inclusive) are available. This data can be sub-divided and packaged into CityLearn compatible datasets to suit the different forecasting methods and analyses that are chosen to be studied.

Initially three example datasets: `train` (2010-2015), `validate` (2016-2017), and `test` (2018-2019); consisting of a set of 6 buildings of similar mean loads are provided.

*Note: the `validate` dataset is a subset of the training data sectioned off to be used for hyper-parameter learning for machine learning methods. However, for classical methods that do not required this, it can be included in the training dataset to extend the available training data.*

<br>

# Using the framework

This codebase provides a framework for integrating forecasting methods into a Linear MPC controller/agent, and applying it to the CityLearn environment. Using a consistent framework allows for the comparability of different approaches taken by different participants. Participants are free to adapt the framework to suit their chosen approach, however **it is important that all participants use the provided configurations of the CityLearn environment (specified by the dataset `schema.json`), so that the performance of different methods of controlling the environment are comparable**.

This initial framework is designed to get teams started and to facilitate discussion around the techniques to be used and compared.  The framework will be updated as necessary and further data will be provided.  The ambition is to mimic the CityLearn competition in that models will be trained on observable data but will then be applied to unseen data for assessment and comparison.

Participants can train their forecasting method/predictor using the training data provided. The model parameters can be tuned using the validation data provided. The fully trained/specified model can then be evaluated on the 'unseen' test data provided.

## Task

The task for participants is to implement a suitable forecasting method(s) for use with the Linear MPC battery scheduling controller, see [Predictor Implementation](#predictor-implementation), and evaluate its performance, see [Evaluation](#evaluation), to gain insight into the role of data in enabling optimised battery scheduling in smart buildings.


## Setup
This project uses Python 3.8. Setup your Python environment for the project using the following commands. (See [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

```
conda create -n myenv python=3.8
pip install -r requirements.txt
```


## Predictor Implementation

Participants should implement their forecasting method/predictor in the `Predictor` class contained in `predictor.py`.

This class *requires* two methods:
1. ```__init__(...)```, which performs any initial setup of the predictor required, such as loading in a trained configuration from a file.
2. ```compute_forecast(self, observations)```,  which implements the forecasting strategy. This method must take in an observation array (as returned from the CityLearn environment, see [Observations](#observations)), and return a set of forecasts for (building loads, solar pv generation powers, grid electricity pricing, grid electricity carbon intensity) as specified in the method docstring.

A dummy forecasting method is provided to illustrate how this class works.

Participants are free to implement additional methods for their `Predictor` class as suits their approach.

For training forecasting strategies, the training dataset is directly available in the `data` directory.

## Forecast Assessment

The quality of forecasting achieved by the implemented prediction method (w.r.t. its match to true/ground-truth values) can be evaluated by running `assess_forecasts.py`, with suitable settings specified within the file.

Please edit the file to assess your forecasts as desired.


## Evaluation

The performance of a given prediction method on the task should be evaluated by running `evaluate.py`, with suitable settings specified within the file,

```
python3 evaluate.py
```

This file sets up the CityLearn environment and Linear MPC controller, and executes the control loop, before evaluating the performance and returning the results.

Please edit the file to log the agent's behaviour as desired.

The price, emission and grid costs are relative to the values achieved if the system is operated without battery storage, i.e. a value of 1 means that the system has the same performance as a system without battery storage. The objective is to minimize an evenly weighted sum of these three contributions.


## System Linear Programming model specification

`linmodel.py` implements the Linear Programming model of the CityLearn environment. Participants are welcomed to have a look at this implementation to see how the model works, however they should not have to interact with this code. A mathematical description of how the model works is available in [this tutorial](https://colab.research.google.com/drive/1Qzs4GhL-OZCY3YoFsVSzA8_BpTnL518P?usp=sharing).

Note: if participants wish to use different objective functions for the Linear MPC controller some adaptation of this file (specifically the `generate_LP` method) and the control loop will be required. Please contact the organisers for help with this.


## Utils

A number of helpful functions for interacting with the CityLearn environment and provided datasets are provided in the `utils` directory.


## Observations

The `observations` data returned by the CityLearn environment when an action is applied consists of a list of observation values for each building in the model, which are indexed as follows:

| Index      | Name | Description | Unit |
| ----------- | ----------- | ----------- | ----------- |
| 0      | Month  | Month of year ranging from 1 (January) through 12 (December).  |-|
| 1      | Day   | Day of week ranging from 1 (Monday) through 7 (Sunday).       |-|
| 2      | Hour    | Hour of day ranging from 1 to 24.       |-|
| 3      | Temperature    | Outdoor dry bulb temperature.       |C|
| 4      | Temperature (Predicted 6h)  | Outdoor dry bulb temperature predicted 6 hours ahead.      |C|
| 5      | Temperature (Predicted 12h)    | Outdoor dry bulb temperature predicted 12 hours ahead.      |C|
| 6      | Temperature (Predicted 24h)    | Outdoor dry bulb temperature predicted 24 hours ahead.      |C|
| 7      | Humidity       | Outdoor relative humidity.      |%|
| 8      | Humidity (Predicted 6h)       | Outdoor relative humidity predicted 6 hours ahead.      |%|
| 9      | Humidity (Predicted 12h)       | Outdoor dry bulb temperature predicted 12 hours ahead.      |%|
| 10      | Humidity (Predicted 24h)       | Outdoor dry bulb temperature predicted 24 hours ahead.      |%|
| 11      | Diffuse Solar       | Diffuse solar irradiance.      |W/m2|
| 12      | Diffuse Solar (Predicted 6h)       | Diffuse solar irradiance predicted 6 hours ahead.      |W/m2|
| 13      | Diffuse Solar (Predicted 12h)       | Diffuse solar irradiance predicted 12 hours ahead.      |W/m2|
| 14      | Diffuse Solar (Predicted 24h)       | Diffuse solar irradiance predicted 24 hours ahead.      |W/m2|
| 15      | Direct Solar       | Direct solar irradiance.      |W/m2|
| 16      | Direct Solar (Predicted 6h)       | Direct solar irradiance predicted 6 hours ahead.      |W/m2|
| 17      | Direct Solar (Predicted 12h)       | Direct solar irradiance predicted 12 hours ahead.      |W/m2|
| 18      | Direct Solar (Predicted 24h)       | Direct solar irradiance predicted 24 hours ahead.      |W/m2|
| 19      | Carbon       | Grid carbon emission rate.      |kgCO2/kWh|
| 20      | Load       |  Total building non-shiftable plug and equipment loads.|kWh    |
| 21      | Solar Generation       | PV electricity generation.     |kWh|
| 22      | Battery Storage       | State of the charge (SOC) of the battery from 0 (no energy stored) to 1 (at full capacity).      |kWh/kWhcapacity|
| 23      | Electricity Consumption       |  Net total building electricity consumption.     |kWh|
| 24      | Electricity Price       | Electricity rate.      |$/kWh|
| 25      | Electricity Price (Predicted 6h)       | Electricity rate predicted 6 hours ahead.      |$/kWh|
| 26      | Electricity Price (Predicted 12h)       | Electricity rate predicted 12 hours ahead.     |$/kWh|
| 27      | Electricity Price (Predicted 24h)       | Electricity rate predicted 24 hours ahead.      |$/kWh|

```python
len(observations) == N # number of buildings in model
len(sublist) == 28 for sublist in observations # observations values as defined above
```

**Note: the prediction observations are provided in order to ensure compatibility of the dataset with the CityLearn environment, however as these are perfect predictions taken from the true data, their use for the development of forecasting methods is considered cheating.**

<br>

## Additional Documentation

[Annex 37 Sub-task A presentation (Feb 16th 2023) slides](https://docs.google.com/presentation/d/1bR0BVOM6U2C5XhC6FKe8YhmrMTmw-Vnx/edit?usp=sharing&ouid=107379212279840102215&rtpof=true&sd=true)


## Further information

[Paper describing development of the CityLearn gym environment](https://arxiv.org/abs/2012.10504)

[CityLearn 2022 challenge winners](https://arxiv.org/abs/2212.01939)
