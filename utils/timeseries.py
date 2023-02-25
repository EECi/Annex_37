"""
Timeseries helper functions.
"""

import numpy as np
from collections.abc import Iterable
import plotly.graph_objects as go
from plotly.offline import plot


def interactive_timeseries_plot(timeseries_list, x_values=None, name_list=None, **kwargs):

    # how do I handle iterables (list of timeseries) vs timeseries to plot?

    fig = go.Figure()

    if all([isinstance(sublist, Iterable) for sublist in timeseries_list]):
        ts_iterable = timeseries_list
    else:
        ts_iterable = [timeseries_list]

    for j,ts in enumerate(ts_iterable):

        # add trace to plot
        viz = True if j == 0 else 'legendonly'
        name = name_list[j] if name_list is not None else j
        fig.add_trace(go.Scatter(
            x = x_values if x_values is not None else np.arange(len(ts)),
            y = ts,
            name = name,
            visible = viz
        ))

    # set plot configuration
    fig.update_layout(
        yaxis_title=kwargs['yaxis_title'] if 'yaxis_title' in kwargs else None,
        xaxis_title=kwargs['xaxis_title'] if 'xaxis_title' in kwargs else None,
        xaxis=dict(rangeslider=dict(visible=True))
        )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    fig.update_yaxes(fixedrange=False) # make y axis dynamic

    fig.update_layout(title_x=0.5) # center title

    plot(
        fig,
        filename=kwargs['filename'] if 'filename' in kwargs else 'temp.html',
        auto_open=False
    )