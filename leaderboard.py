import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


save_html = False
results_file = 'forecast_results.csv'
results_file = os.path.join('outputs', results_file)

data = pd.read_csv(results_file)


# create plot
model_names = data['Model'].values
traces = []
for model_name in model_names:
    trace = go.Scatter(
        x=data.columns[1:],
        y=data[data['Model'] == model_name].values.tolist()[0][1:],
        name=model_name,
        mode='markers',
        hovertext=[model_name] * len(data.columns[1:]),
        hoverinfo='text',
        marker=dict(
            size=10,
            opacity=0.8,
            line=dict(width=0.5, color='white')
        )
    )
    traces.append(trace)

layout = go.Layout(
    title='Leaderboard',
    xaxis=dict(title='Metrics'),
    yaxis=dict(title='Globally Normalised Mean Absolute Error'),
    legend=dict(
        title='Models',
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="left",
        x=0
    ),
)
fig_scatter = go.Figure(data=traces, layout=layout)

best_tally = []
for column in data.columns[1:]:
    data[column] = data[column].rank(method='min').astype(int)
    best = data.nsmallest(1, column)[column].index
    data.loc[best, column] = data.loc[best, column].astype(str) + ' 🥇'
    best_tally.append(best.item())
trace_table = go.Table(
    header=dict(values=list(data.columns)),
    cells=dict(values=[data[col] for col in data.columns])
)

fig_table = go.Figure(data=[trace_table])
fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'table'}], [{'type': 'scatter'}]],
                    vertical_spacing=0,
                    horizontal_spacing=0,
                    row_heights=[0.4, 0.6])


fig.add_trace(trace_table, row=1, col=1)
for trace in traces:
    fig.add_trace(trace, row=2, col=1)

fig.update_layout(
    title='Leaderboard',
    xaxis=dict(title='Dataset'),
    yaxis=dict(title='Globally Normalised Mean Absolute Error'),
    legend=dict(
        title='Models',
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="left",
        x=0
    ),
    height=800
)

fig.show()
if save_html:
    pio.write_html(fig, file='outputs/leaderboard.html', auto_open=True)

max(set(best_tally), key=best_tally.count)
with open('outputs/leaderboard.md', 'w', encoding='utf-8') as f:
    f.write('# Leaderboard\n')

    f.write('### Forecast \n')
    f.write(f'Best Model = {data.iloc[1]["Model"]}\n')
    f.write('\n')
    f.write(data.to_html(index=False))
