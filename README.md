
# Annex 37
## Smart design and control of energy storage systems 

The aim of [Annex 37](https://iea-es.org/task-37/) is to investigate smart design and control strategies for energy storage systems at both the supply side and demand side. As part of Sub-task A - Demand and Supply Prediction - we aim to compare techniques for working with real-time demand data for optimisation of battery storage.  This repository contains data and a template for a reinforcement learning approach to be used for comparison across techniques.

# The CityLearn 2022 competition

The approach is directly based on the [CityLearn 2022 challenge](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge) run as part of [NeurIPS 2022](https://nips.cc).  In the challenge, international teams competed to minimise costs and carbon emissions across a group of residential buildings equipped with solar PV and battery storage. The teams were able to use rule-based controllers, model predictive controllers or reinforcement learning with single agent or multi agent set-up.  The competition used [CityLearn](https://github.com/intelligent-environments-lab/CityLearn/tree/citylearn_2022), an open source OpenAI gym environment or the implementation of multi-agent reinforcement learning for building energy coordination and demand response in cities.

In the first phase of the competition, demand data for 5 buildings were provided and teams were able to train their models on the supplied data.  In phase 2, the models were applied to an unseen dataset for another 5 buildings, and in phase 3 to an unseen dataset for a final 7 buildings.  The final ranking was calculated based on the performance over all 3 datasets.

Final performance was judged on three criteria:

1)  minimum normalised cost of electricity taken from the grid supply,
2)  minimum normalised carbon emissions, and
3)  minimum [grid cost](https://discourse.aicrowd.com/t/announcement-update-to-evaluation-and-leaderboard/8156).

The minimum grid cost is the average of the normalised ramping, $R$ and normalised (1-load factor, $L$), where ramping is the smoothness of the district's load profile i.e. low ramping implies a gradual increase in grid electricity demand even after self-generation from the PV becomes unavailable, and high ramping implies abrupt changes that could lead to unacceptable strain on the grid infrastructure and supply.  $R$ is calculated as the sum of the absolute difference of net electricity consumption between consecutive timesteps. The load factor, $L$ is the efficiency of electricity consumption, calculated as the average ratio of monthly average and monthly maximum net electricity consumption.  

The results were all normalised with respect to the values with no operational battery.

# Overview
[comment]: <> (Todo: add some comment aboutf MPC format, currently mentions reinforment learning format.)
For Annex 37, we wish to assess the range of battery storage control from unsupervised to supervised learning approaches.  Using the CityLearn gym we are able to compare different approaches using a consistent framework and thereby ensure comparability. The template is given in a reinforcement learning format i.e. an agent learns to perform an action so as to maximise a reward, however Annex participants are free to explore different approaches and to adapt the template as necessary. What is important is that we all use the same data and the same battery configuration.

[comment]: <> (Todo: add some comment about the new dataset, currently 5 buildings.)
This repository contains a simplified version of the competition code, together with data for 5 buildings as issued for the competition phase 1.  The results of an evaluation are given in terms of the average price cost, average emission cost and average grid cost, all normalised against the case with no battery storage i.e. values greater than 1 suggest worse performance than the no-battery case.

This initial framework is designed to get teams started and to facilitate discussion around the techniques to be used and compared.  The framework will be updated as necessary and further data will be provided.  The ambition is to mimic the CityLearn competition in that models will be trained on observable data but will then be applied to unseen data for assessment and comparison.

[comment]: <> (Todo: add some comment about the training, validation and test split. Below is a placeholder.)
Teams are free to use any method to train a predictor using the training data provided. The model parameters can be 
tuned use the validation set. The trained model can then be evaluated on the unseen test set by running the 
`evaluation.py` file.

# Using the framework
The following sections outline how to use the framework in a reinforcement learning strategy.

## Setup
This project uses Python 3.8.
```
pip install -r requirements.txt
```

## Method implementation
Implement your method in the `compute_action` function in the `agent.py` file:

```python
def compute_action(self, observation):
    """
    Below is the suggested format for this function:
    - Use the observation to compute the reward.
    - Update the agent parameters using the reward.
    - Compute the action for the observation

    Inputs:
        observation - List of observations from the env: observation[building_index, observation_index]
    Returns:
        actions - List of actions: actions[building_index]
            eg. for 5 buildings each with action 0.5 the format is as follows [[0.5], [0.5], [0.5], [0.5], [0.5]]
            Actions are between 0 and 1.

    Please make sure the action for each building is in the same order as the observations for each building.

    """
    assert self.num_buildings is not None

    # Define you reward here
    # ==============================================================================================================
    rewards = [0 for _ in range(self.num_buildings)]
    # ==============================================================================================================

    # Update your agent parameters here
    # ==============================================================================================================

    # ==============================================================================================================

    # Compute your action here
    # ==============================================================================================================
    actions = [[0] for _ in range(self.num_buildings)]
    # ==============================================================================================================
    return actions
```

 

To track variables you may want to define attributes with the `__init__` function:
```python
    def __init__(self):
        self.num_buildings = None
        self.action_space = None

        # You may want to track some variables, eg. observations or actions from previous time steps. Do so here.
        # ==============================================================================================================
        self.prev_observation = None
        # ==============================================================================================================
```

Feel free to add attributes and methods to the `Agent` class as required.  You may also want to implement the agent 
resetting behaviour in the `register_reset` function.

## Evaluation
Select the number of episodes you want to train your agent for, and the dataset you want to use by editing the 
`Constants` class in the `local_evaluation.py` file.

```python
class Constants:
    episodes = 1
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
```

Please edit the file to log the agent's behaviour as desired. 

The price, emission and grid costs are relative to the 
value for the system without battery storage i.e., a value of 1 means that the system has the same performance as a system without battery storage. The lower the cost, the better.

## Observations

This is a key to observation identifiers used in the gym environment.

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

## Further information

[Paper describing development of the CityLearn gym environment](https://arxiv.org/abs/2012.10504)

[The challenge winners](https://arxiv.org/abs/2212.01939)

