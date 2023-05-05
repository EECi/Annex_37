"""Perform model inference & plot dynamic timeseries of results."""

import os
import json
import numpy as np

import warnings

from models import TFT_Predictor, NHiTS_Predictor, DeepAR_Predictor, LSTM_Predictor, GRU_Predictor

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import plot as ply_plot

def main(
        model_group_name, model_architecture, predictor_model,
        model_type, model_name, building_index,
        dataset_path, idx_start, idx_end,
        save_path='temp.html', **kwargs
    ):

    # define model architecture handling
    attention_models = ['TFT']
    no_loss_plot_models = ['DeepAR']
    unsupported_models = ['RNN']

    if model_architecture in unsupported_models:
        # TODO: add support for RNN model architecture
        # NOTE: issue is that predict only returns a single timeseries not quantiles
        raise ValueError(f"Model architecture {model_architecture} not yet supported.")

    # load model group - note ordering does not matter
    model_group = predictor_model(model_group_name, load='group')

    # check requested model is valid
    if model_type in ['load','solar']:
        assert model_name in model_group.model_names[model_type], f"{model_name} is not a valid {model_type} model in model group '{model_group_name}'."
    elif model_type in ['pricing','carbon']:
        assert model_name == model_group.model_names[model_type], f"{model_name} is not a valid {model_type} model in model group '{model_group_name}'."
    else:
        raise ValueError("`model_type` argument must be one of ('load','solar','pricing','carbon').")
    
    # grab requested model model
    model = model_group.models[model_type][model_name]

    # construct inference dataset
    batch_size = 128
    n_workers = min(os.cpu_count(),4)
    dataset_ds, = model_group.format_CityLearn_datasets([dataset_path], model_type=model_type, model_name=model_name, building_index=building_index)
    dataset_dl = dataset_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=n_workers)

    # perform inference
    print("\nPerforming inference...")
    quantile_predictions, x = model.predict(dataset_dl, mode="quantiles", mode_kwargs={'quantiles':[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]}, return_x=True)
    if model_architecture not in no_loss_plot_models: 
        raw_predictions, x = model.predict(dataset_dl, mode="raw", return_x=True)
    print(f"Loss: {round(model.loss.compute().item(),4)}")

    # Plot dynamic inference timeseries chart
    print("Producing plot...")
    # ========================================================================
    max_encoder_length = model_group.L
    max_prediction_length = model_group.T
    val_max = kwargs['val_max'] if 'val_max' in list(kwargs.keys()) else 1.25*np.amax(x['decoder_target'].detach().cpu().numpy())
    max_attention = 0.4
    transition_duration = 0
    frame_duration = 100

    val_titles = {
        'load': "Electrical Load (kWh)",
        'solar': "Solar generation (kWh)",
        'pricing': "Electricity price (£/kWh)",
        'carbon': "Carbon intensity (kg_CO2/kWh)"
    }
    val_title = val_titles[model_type]

    prediction_losses = []

    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [-1*(max_encoder_length+1), max_prediction_length+2], "showgrid":True}
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

        # get encoder length, compute attention
        encoder_length = x["encoder_lengths"][0]
        if model_architecture in attention_models:
            interpretation = model.interpret_output(raw_predictions.iget(slice(idx, idx + 1)))

        if model_architecture not in no_loss_plot_models:
            # outrageously hacky way of computing prediction loss
            fig, dummy_ax = plt.subplots()
            model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True, ax=dummy_ax)
            prediction_loss = float(dummy_ax.get_title().split(' ')[1])
            plt.close(fig)
            del fig, dummy_ax
            prediction_losses.append(prediction_loss)

        # construct plots for each frame
        plot_data = [
            go.Scatter( # p95 area fill
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,0],
                mode="lines",
                line=dict(color="rgba(249, 115, 6, 0.1)", width=0),
                name="p95",
                showlegend=False
            ),
            go.Scatter(
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,6],
                mode="lines",
                line=dict(color="rgba(249, 115, 6, 0.1)", width=0),
                name="p95",
                fill='tonexty'
            ),
            go.Scatter( # p80 area fill
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,1],
                mode="lines",
                line=dict(color="rgba(249, 115, 6, 0.2)", width=0),
                name="p80",
                showlegend=False
            ),
            go.Scatter(
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,5],
                mode="lines",
                line=dict(color="rgba(249, 115, 6, 0.2)", width=0),
                name="p80",
                fill='tonexty'
            ),
            go.Scatter( # p50 area fill
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,2],
                mode="lines",
                line=dict(color="rgba(249, 115, 6, 0.4)", width=0),
                name="p50",
                showlegend=False
            ),
            go.Scatter(
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,4],
                mode="lines",
                line=dict(color="rgba(249, 115, 6, 0.4)", width=0),
                name="p50",
                fill='tonexty'
            ),

            go.Scatter( # decoder target
                x=list(range(max_prediction_length)),
                y=x['decoder_target'][idx],
                mode="lines",
                line=dict(color="rgba(1, 21, 62, 1)", width=2),
                legendgroup="target",
                name="Target"
            ),
            go.Scatter( # encoder target
                x=[-1*i for i in reversed(range(1,encoder_length+1))],
                y=x['encoder_target'][idx],
                mode="lines",
                line=dict(color="rgba(1, 21, 62, 1)", width=2),
                legendgroup="target",
                showlegend=False,
                name="Target"
            ),

            go.Scatter( # mean prediction
                x=list(range(max_prediction_length)),
                y=quantile_predictions[idx,:,3],
                mode="lines",
                line=dict(color="rgba(117,187,253, 1)", width=2),
                name="Mean prediction"
            ),
        ]

        if model_architecture not in no_loss_plot_models:
            plot_data.extend([
            go.Bar( # prediction loss bar chart
                x=[max_prediction_length+1],
                y=[prediction_loss],
                width=1,
                marker=dict(color="rgba(244, 50, 12, 1)"),
                name="Loss",
                yaxis='y2',
            )])

        if model_architecture in attention_models:
            plot_data.extend([
            go.Scatter( # encoder attention
                x=[-1*i for i in reversed(range(1,encoder_length+1))],
                y=interpretation['attention'][0,-encoder_length:],
                mode="lines",
                line=dict(color="rgba(63, 155, 11, 0.5)", width=2),
                yaxis='y3',
                name="Attention"
            )])

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
            title=val_title,
            range=[0,val_max],
            side="left"
        ),
        yaxis2=dict(
            title="Prediction Loss",
            range=[0,2.5*np.mean(prediction_losses)],
            anchor="x",
            overlaying="y",
            side="right",
            tickmode="sync"
        ),
        yaxis3=dict(
            title="Attention",
            range=[0,max_attention],
            anchor="free",
            overlaying="y",
            #side="left",
            showgrid=False,
            autoshift=True
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
    # ========================================================================


if __name__ == '__main__':

    # set paths for train & validation datasets for training
    dataset_path = os.path.join('data','example','validate')

    # grab building ids in specified dataset
    with open(os.path.join(dataset_path,'metadata_ext.json')) as json_file:
        UCam_ids = json.load(json_file)["UCam_building_ids"]

    # specify model to be used for inference
    model_group_name = 'test-GRU'
    model_architecture = 'RNN'
    predictor_model = GRU_Predictor

    model_type = 'load'
    model_name = f'load_{UCam_ids[0]}'
    building_index = 0

    # specify scope of inference
    duration = 24*7*12
    idx_start = 0
    idx_end = idx_start+duration

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore',module=r'pytorch_lightning')

        main(
            model_group_name, model_architecture, predictor_model,
            model_type, model_name, building_index,
            dataset_path, idx_start, idx_end,
            val_max=300
        ) # val_max=300