"""Test DMD forecasting and plot."""

#!/usr/bin/env python
"""
Assess performance of predictor model forecasts.

Perform prediction inference using given predictor model with
specified dataset to evaluate predictor forecasting performance
in comparison to ground truth values of prediction variables.
"""

import os
import sys
import csv
import time
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot as ply_plot

from citylearn.citylearn import CityLearnEnv
from models import DMDPredictor

def forecast(predictor, schema_path, tau, T, **kwargs):

    print("Starting forecasting.")

    # Initialise CityLearn environment object.
    env = CityLearnEnv(schema=schema_path)

    # Initialise Predictor object.
    predictor.initialise_forecasting(env)

    # Initialise logging objects.
    load_logs = {b.name:{'forecasts': [], 'actuals': []} for b in env.buildings}
    pv_gen_logs = {'forecasts': [], 'actuals': []}
    pricing_logs = {'forecasts': [], 'actuals': []}
    carbon_logs = {'forecasts': [], 'actuals': []}

    # Initialise forecasting loop.
    forecast_time_elapsed = 0
    num_steps = 0
    done = False

    observations = env.reset()

    # Execute control loop.
    with tqdm(total=env.time_steps) as pbar:

        while (not done) and (num_steps < T):
            if num_steps % 100 == 0:
                pbar.update(100)

            # Set up custom data input for method.
            forecast_kwargs = {}
            forecast_kwargs['t'] = env.time_step

            # Compute forecast.
            forecast_start = time.perf_counter()
            forecasts = predictor.compute_forecast(observations, **forecast_kwargs)
            forecast_time_elapsed += time.perf_counter() - forecast_start

            # Perform logging.
            if forecasts is None:   # forecastor opt out
                pass    # no forecast to evaluate
            else:
                # Log forecasts.
                for i, b in enumerate(env.buildings):
                    load_logs[b.name]['forecasts'].append(forecasts[0][i].reshape(-1))
                pv_gen_logs['forecasts'].append(forecasts[1].reshape(-1))
                pricing_logs['forecasts'].append(forecasts[2].reshape(-1))
                carbon_logs['forecasts'].append(forecasts[3].reshape(-1))
                # Log ground-truth values.
                # note abuse of Python array slicing to give variable length actuals toward end of lists
                for i,b in enumerate(env.buildings):
                    load_logs[b.name]['actuals'].append(b.energy_simulation.non_shiftable_load[env.time_step+1:env.time_step+1+tau])
                pv_gen_logs['actuals'].append(b.energy_simulation.solar_generation[env.time_step+1:env.time_step+1+tau])
                pricing_logs['actuals'].append(b.pricing.electricity_pricing[env.time_step+1:env.time_step+1+tau])
                carbon_logs['actuals'].append(b.carbon_intensity.carbon_intensity[env.time_step+1:env.time_step+1+tau])

            # Step environment.
            actions = np.zeros((len(env.buildings), 1))
            observations, _, done, _ = env.step(actions)

            num_steps += 1

    print("Forecasting complete.")

    return [load_logs, pv_gen_logs, pricing_logs, carbon_logs]


def interactive_plot(variable_logs, variable_name, tau,
                     idx_start, idx_end, val_max,
                     save_path='temp.html', **kwargs):

    transition_duration = 0
    frame_duration = 100

    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [0, tau], "showgrid":True}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": frame_duration, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {"duration": transition_duration, "easing": "linear"}
                                    }],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}
                                        }],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 12},
            "prefix": "Timestep: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": transition_duration, "easing": "linear"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make frames
    for j,idx in enumerate(range(idx_start,idx_end+1)):

        frame = {"data": [], "name": str(idx)}

        # construct plots for each frame
        plot_data = [
            go.Scatter( # ground truth value
                x=list(range(tau)),
                y=variable_logs['actuals'][idx],
                mode="lines",
                line=dict(color="rgba(1, 21, 62, 1)", width=2),
                legendgroup="target",
                name="Target"
            ),

            go.Scatter( # mean prediction
                x=list(range(tau)),
                y=variable_logs['forecasts'][idx],
                mode="lines",
                line=dict(color="rgba(117,187,253, 1)", width=2),
                name="Prediction"
            ),
        ]

        if j == 0: # set initial data to plot
            fig_dict["data"] = plot_data

        frame = go.Frame(
            data=plot_data,
            name=str(idx),
        )

        fig_dict["frames"].append(frame)

        slider_step = {"args": [
            [idx],
            {"frame": {"duration": transition_duration, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": transition_duration},
            }
            ],
            "label": idx,
            "method": "animate"
        }
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    fig.update_layout(
        yaxis=dict(
            title='DMD prediction - %s'%variable_name,
            range=[0,val_max],
            side="left"
        ),
        legend=dict(
            traceorder="reversed",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    print("Rendering plot...")
    ply_plot(
        fig,
        filename=save_path,
        auto_open=False,
        auto_play=False
    )

    fig.show()



if __name__ == '__main__':
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', module=r'pydmd')

        UCam_ids = [0,3,9,11,12,15,16,25,26,32,38,44,45,48,49] # set as list of same int to test model on different buildings
        UCam_ids = [5,11,14,16,24,29]

        tau = 48  # model prediction horizon (number of timesteps of data predicted)
        T = 24*7*4 # number of timesteps to forecast for
        dataset_dir = os.path.join('example', 'test')  # dataset directory
        schema_path = os.path.join('data', dataset_dir, 'schema.json')

        print("Initialising predictor.")
        predictor = DMDPredictor(building_indices=UCam_ids, dataset_dir=os.path.join('data','example'))

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', module=r'cvxpy')
            logs = forecast(predictor, schema_path, tau, T)

        # TODO: plot forecasts for inspection

        duration = 24*7*3
        idx_start = 0
        idx_end = idx_start+duration

        plot_variable = 'load'
        variable_logs =  logs[0][list(logs[0].keys())[2]] #logs[1]

        interactive_plot(variable_logs, plot_variable, tau, idx_start, idx_end, val_max=1000)
