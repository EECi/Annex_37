#!/usr/bin/env python
"""
Assess performance of predictor model forecasts.

Perform prediction inference using given predictor model with
specified dataset to evaluate predictor forecasting performance
in comparison to ground truth values of prediction variables.
"""

import os
import time
import numpy as np
import cvxpy as cp
import pandas as pd

from tqdm import tqdm
from citylearn.citylearn import CityLearnEnv
from predictor import Predictor
import matplotlib.pyplot as plt


def compute_metric_score(forecasts_array, ground_truth_array, metric, global_mean_norm=False):
    """Compute mean metric score over set of forecasts corresponding
    to ground truth arrays for specified metric function.

    Args:
        forecasts_array (List[List]): list of forecast list to compute metric values for.
        ground_truth_array (List[List]): list of ground truth value lists corresponding to
        forecasts (length can be less than or equal to corresponding forecast length).
        metric (function): function computing desired forecast performance metric
        for a given forecast and corresponding ground truth (taken as `np.array`s).
        mean_norm (bool, optional): Whether to normalise the mean metric value by the
        mean of the underlying ground truth timeseries. Defaults to False.

    Returns:
        metric_score (float): mean metric score over set of forecasts & ground truths.
    """

    assert len(forecasts_array) == len(ground_truth_array), "Must provide same number of forecasts and ground truths to compare."

    metric_scores = []

    for forecast, actual in zip(forecasts_array, ground_truth_array):
        a = np.array(actual)
        f = np.array(forecast)[:len(a)]
        metric_scores.append(metric(f,a))

    metric_score = np.mean(metric_scores)

    if global_mean_norm:
        metric_score = metric_score/np.mean([l[0] for l in ground_truth_array if len(l) > 0])

    return metric_score

def MAE(prediction, actual):
    return np.mean(np.abs((prediction-actual)))

def RMSE(prediction, actual):
    return np.sqrt(np.mean(np.power(prediction-actual,2)))


def assess(schema_path, tau, building_breakdown=False, **kwargs):
    """Evaluate forecasting performance of given Predictor model for
    dataset specified by provided schema.

    Args:
        schema_path (Str or os.Path): path to schema defining simulation data.
        tau (int): length of planning horizon
        building_breakdown (bool): indicator for whether building resolved
        performance metric values are reported. Defaults to 'False'.

    Returns:
        results (dict): dictionary containing performance metrics from forecasting
        assessment, and forecasting time.
    """

    print("Starting assessment.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Predictor object.

    # ========================================================================
    # insert your import & setup code for your predictor here.
    # ========================================================================

    predictor = Predictor(len(env.buildings), tau)

    # Initialise logging objects.
    load_logs = {b.name:{'forecasts':[], 'actuals':[]} for b in env.buildings}
    pv_gen_logs = {b.name:{'forecasts':[], 'actuals':[]} for b in env.buildings}
    pricing_logs = {'forecasts':[], 'actuals':[]}
    carbon_logs = {'forecasts':[], 'actuals':[]}


    # Initialise forecasting loop.
    forecast_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()
    # print ('observe env \n', observations)

    fig, ax = plt.subplots(figsize=(4, 4))
    # fig2, ax2 = plt.subplots(figsize=(12, 3))

    # Execute control loop.
    with tqdm(total=env.time_steps) as pbar:
        n=0
        k=0
        while not done:
            print ('time step = ', num_steps)
            if num_steps%100 == 0:
                pbar.update(100)

            # Compute forecast.

            forecast_start = time.perf_counter()

            if num_steps % tau != 0:
                '''
                set compute_forecast to False, and slide window across forecast buffer
                '''

                # observations_inputs = []
                # for building in env.buildings:
                #     observations_inputs.append(building.weather.diffuse_solar_irradiance[
                #                       env.time_step + 1:env.time_step + 1 + tau*2])


                # call compute_forecast with False compute flag, to update buffer but not return a forecast
                print ('observations \n', observations)
                predictor.compute_forecast(env=CityLearnEnv, observations=observations, t=env.time_step, compute_forecast=False)

                # retrieve next tau forecast from stored forecast in buffer
                # need to update self.buffer by one timestep
                forecasts = []

                for i,_ in enumerate(['load','solar']):
                    forecasts.append([predictor.forecasts_buffer[i][0][n:tau+k], predictor.forecasts_buffer[i][1][n:tau+k]])
                forecasts.append(predictor.forecasts_buffer[2][n:tau + k])
                forecasts.append(predictor.forecasts_buffer[3][n:tau + k])

                """
                Plot selected forecasted signals for debugging within control loop
                """
                
                ax.cla()
                ax.plot(pd.Series(range(n,tau+k)), forecasts[0][1],c='r', label='load_1 forecast')
                ax.plot(pd.Series(range(n,tau+k)), env.buildings[0].energy_simulation.non_shiftable_load[env.time_step + 1:env.time_step + 1 + tau],
                         c='k', label='load_1 actual')
                # ax.plot(pd.Series(range(n,tau+k)), predictor.forecasts_buffer[1][0][n:tau+k], c='r', label='solar_0 forecast')
                # ax.plot(pd.Series(range(n,tau+k)), env.buildings[1].energy_simulation.solar_generation[num_steps:num_steps + tau],
                #          c='k', label='solar_0 actual')
                ax.set_title('48 hour forecast at time= '+str(num_steps))
                ax.set_ylim (0, 1000)
                ax.legend()
                plt.pause(0.002)

                # if num_steps == 812:
                #     input("Press Enter to continue...")

                n+=1
                k+=1

            else:
                '''
                compute forecast for the next tau*2 hours and store in forecast buffer
                '''
                # set sliding counter to zero
                n = 0
                k = 0

                # initialise past control input attributes (solar diffuse and dir data)
                predictor.initialise_forecasting(env)

                # call compute_forcast with each timestep to fit HODMD or DMDC on previous L timesteps (returns tau*2 hours)
                forecasts_buffer, reconstructed = predictor.compute_forecast(env=CityLearnEnv, observations=observations, t=env.time_step, compute_forecast=True)

                # store forecast to the predictor's forecast buffer
                predictor.forecasts_buffer = forecasts_buffer

                # select only tau timesteps from the tau*2 forecast in buffer
                forecasts = []
                for i,_ in enumerate(['load','solar']):
                    forecasts.append([predictor.forecasts_buffer[i][0][n:tau+k], predictor.forecasts_buffer[i][1][n:tau+k]])
                    # forecasts.append(predictor.forecasts_buffer[0][i][n:tau+k])
                    # forecasts.append(predictor.forecasts_buffer[1][i][n:tau+k])
                forecasts.append(predictor.forecasts_buffer[2][n:tau + k])
                forecasts.append(predictor.forecasts_buffer[3][n:tau + k])

                #
                # fig, ax = plt.subplots()
                # observed_test = np.array([env.buildings[0].energy_simulation.solar_generation[num_steps:num_steps + tau * 2]])
                #
                # plt.ion()
                # # ax.plot(env.buildings[0].energy_simulation.solar_generation[env.time_step+1:env.time_step+1+tau], label='observed tr ')
                # ax.plot(reconstructed, label='reconstructed ')
                # ax.plot(np.arange(len(reconstructed), len(reconstructed) + len(forecasts_buffer[1][0]))[:48], forecasts[1][0],
                #         label='forecast ')
                #
                # ax.plot(np.arange(len(reconstructed), len(reconstructed) + len(forecasts_buffer[1][0])), observed_test.reshape(96, ),
                #         label='observed test ' )
                # ax.legend()
                # plt.pause(1)
                # input("Press Enter to continue...")

                # for f in predictor.forecasts_buffer:
                #     forecasts.append(f[n:tau+k])

                """

                ax2.set_title(('96 hour buffer forecast at time= '+str(num_steps)))

                # ax2.plot(pd.Series(range(n,tau)), predictor.forecasts_buffer[0][0][:tau], label='load_0 forecast')
                # ax2.plot(pd.Series(range(n,tau)),env.buildings[0].energy_simulation.non_shiftable_load[env.time_step + 1:env.time_step + 1 + tau],
                #          label='load_0 actual')

                ax2.plot(pd.Series(range(n,tau)), forecasts[1][0], label='solar_0 forecast')
                ax2.plot(pd.Series(range(n,tau)), env.buildings[0].energy_simulation.solar_generation[env.time_step + 1:env.time_step + 1 + tau],
                          label='solar_0 actual')
                ax2.legend()
                plt.pause(1)
                input("Press Enter to continue...")
                """

            forecast_time_elapsed += time.perf_counter() - forecast_start

            # Perform logging.
            if forecasts is None: # forecaster opt out
                pass # no forecast to evaluate
            else:
                # Log forecasts.
                for i,b in enumerate(env.buildings):
                    load_logs[b.name]['forecasts'].append(forecasts[0][i])
                    pv_gen_logs[b.name]['forecasts'].append(forecasts[1][i])
                    # ax.plot(forecasts[0][i], label = 'load_f')
                    # ax.plot(b.energy_simulation.non_shiftable_load[env.time_step+1:env.time_step+1+tau], label = 'load_a' )
                    # ax.plot(forecasts[1][i], label = 'solar_f')
                    # ax.plot(b.energy_simulation.solar_generation[env.time_step+1:env.time_step+1+tau], label='solar_a')
                pricing_logs['forecasts'].append(forecasts[2])
                carbon_logs['forecasts'].append(forecasts[3])
                # ax.plot(forecasts[2], label = 'price_f')
                # ax.plot(b.pricing.electricity_pricing[env.time_step+1:env.time_step+1+tau], label='price_a')
                # ax.plot(forecasts[3], label = 'carbon_f')
                # ax.plot(b.carbon_intensity.carbon_intensity[env.time_step+1:env.time_step+1+tau], label='carbon_a')

            # Log ground-truth values.
            # note abuse of Python array slicing to give variable length actuals toward end of lists
            for i,b in enumerate(env.buildings):
                load_logs[b.name]['actuals'].append(b.energy_simulation.non_shiftable_load[env.time_step+1:env.time_step+1+tau])
                pv_gen_logs[b.name]['actuals'].append(b.energy_simulation.solar_generation[env.time_step+1:env.time_step+1+tau])
            pricing_logs['actuals'].append(b.pricing.electricity_pricing[env.time_step+1:env.time_step+1+tau])
            carbon_logs['actuals'].append(b.carbon_intensity.carbon_intensity[env.time_step+1:env.time_step+1+tau])

            # Step environment.
            actions = np.zeros((len(env.buildings),1))
            observations, _, done, _ = env.step(actions)
            num_steps += 1

    print("Assessment complete.")

    # Compute forecasting performance metrics.
    metrics = [MAE,RMSE]
    metric_names = ['gmnMAE','gmnRMSE']
    globally_mean_normalised = [True,True]

    load_metrics = {
        b.name:{
            mname: compute_metric_score(load_logs[b.name]['forecasts'],load_logs[b.name]['actuals'],metric,gnorm)\
                for metric,mname,gnorm in zip(metrics,metric_names,globally_mean_normalised)
        } for b in env.buildings
    }
    load_metrics['buildings_average'] = {mname: np.mean([load_metrics[b.name][mname] for b in env.buildings])\
        for mname in metric_names}

    pv_gen_metrics = {
        b.name:{
            mname: compute_metric_score(pv_gen_logs[b.name]['forecasts'],pv_gen_logs[b.name]['actuals'],metric,gnorm)\
                for metric,mname,gnorm in zip(metrics,metric_names,globally_mean_normalised)
        } for b in env.buildings
    }
    pv_gen_metrics['buildings_average'] = {mname: np.mean([pv_gen_metrics[b.name][mname] for b in env.buildings])\
        for mname in metric_names}

    pricing_metrics = {
            mname: compute_metric_score(pricing_logs['forecasts'],pricing_logs['actuals'],metric)\
                for metric,mname in zip(metrics,metric_names)
        }

    carbon_metrics = {
            mname: compute_metric_score(carbon_logs['forecasts'],carbon_logs['actuals'],metric)\
                for metric,mname in zip(metrics,metric_names)
        }

    print("=========================Results=========================")
    print(f"Total time taken for forecasting: {round(forecast_time_elapsed,1)}s")
    print("")
    print("=====Buildings Average=====")
    print("---Load---")
    for mname in metric_names: print(f"{mname}: {round(load_metrics['buildings_average'][mname],5)}")
    print("---Solar Generation---")
    for mname in metric_names: print(f"{mname}: {round(pv_gen_metrics['buildings_average'][mname],5)}")
    print("=====Pricing=====")
    for mname in metric_names: print(f"{mname}: {round(pricing_metrics[mname],5)}")
    print("=====Carbon Intensity=====")
    for mname in metric_names: print(f"{mname}: {round(carbon_metrics[mname],5)}")
    print("")
    if building_breakdown:
        for b in env.buildings:
            print(f"====={b.name}=====")
            print("---Load---")
            for mname in metric_names: print(f"{mname}: {round(load_metrics[b.name][mname],5)}")
            print("---Solar Generation---")
            for mname in metric_names: print(f"{mname}: {round(pv_gen_metrics[b.name][mname],5)}")


    results = {
        'Load Forecasts': load_metrics,
        'Solar Generation Forecasts': pv_gen_metrics,
        'Pricing Forecasts': pricing_metrics,
        'Carbon Intensity Forecasts': carbon_metrics,
        'Forecast Time': forecast_time_elapsed
    }

    return results


if __name__ == '__main__':
    import warnings

    dataset_dir = os.path.join('example','test') # dataset directory

    schema_path = os.path.join('data',dataset_dir,'schema.json')

    tau = 48 # model prediction horizon (number of timesteps of data predicted)

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',module=r'cvxpy')

        results = assess(schema_path, tau, building_breakdown=True)
