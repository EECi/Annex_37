import time
import numpy as np

"""
Please add code to log your agent's performance as desired.
"""

from predictor import Predictor
from citylearn.citylearn import CityLearnEnv
from utils import env_reset


def evaluate(schema_path):
    print("Starting evaluation")

    # Instantiate objects.
    env = CityLearnEnv(schema=schema_path)
    obs_dict = env_reset(env)
    predictor = Predictor()
    actions = predictor.register_reset(obs_dict)

    # Set up the loop.
    agent_time_elapsed = 0
    num_steps = 0
    done = False
    while not done:
        observations, _, done, _ = env.step(actions)
        step_start = time.perf_counter()
        actions = predictor.compute_action(observations)        # todo: remove
        # forecast = predictor.compute_forecast(observations)   # todo: add
        # actions = LMPC.get_actions(forecast)                  # todo: add
        agent_time_elapsed += time.perf_counter() - step_start

        num_steps += 1
        if num_steps % 1000 == 0:
            print(f"Num Steps: {num_steps}")

    metrics = env.evaluate()  # Provides a break down of other metrics that might be of interest.
    if np.any(np.isnan(metrics['value'])):
        raise ValueError("Some of the metrics returned are NaN, please contant organizers.")

    print("=========================Completed=========================")
    print(f"Price Cost: {metrics.iloc[5].value}")
    print(f"Emission Cost: {metrics.iloc[2].value}")
    print(f"Grid Cost:{np.mean([metrics.iloc[0].value, metrics.iloc[6].value])}")
    print(f"Total time taken by agent: {agent_time_elapsed}s")
    

if __name__ == '__main__':
    # todo: remove the citylearn_challenge_2022_phase_1 dataset from the data folder
    # todo: add the new test dataset to the data folder
    # todo: change the schema path to the test dataset.
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
    evaluate(schema_path)
