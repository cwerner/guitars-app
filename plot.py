###### Plotting

import json
import pandas as pd
import plotly
import plotly.graph_objs as go


def prediction_barchart(result, class_labels, class_dict=None):

    # data is list of name, value pairs
    y_values, x_values = map(list, zip(*result))
    # Create the Plotly Data Structure

    x_values = [x  if x < 0 else x for x in x_values]
    y_values = class_labels
    if class_dict:
        y_values = [class_dict[y] for y in y_values]

    # classify based on prob.
    labels = ['Hm?', 'Maybe', 'Probably', 'Trust me']
    cols   = ['red', 'orange', 'lightgreen', 'darkgreen']

    colors = dict(zip(labels, cols))
  
    
    bins = [-0.001, 10, 25, 75, 100.001]

    # Build dataframe
    df = pd.DataFrame({'y': y_values,
                       'x': x_values,
                       'label': pd.cut(x_values, bins=bins, labels=labels)})

    bars = []
    for label, label_df in df.groupby('label'):
        bars.append(go.Bar(x=label_df.x[::-1],
                           y=label_df.y[::-1],
                           name=label,
                           marker={'color': colors[label]},
                           orientation='h'))

    graph = dict(
        data=bars,
        layout=dict(

            #title='Bar Plot',
            xaxis=dict(
                title="Probability",
                range=[0, 100]
                ),

            hovermode='y',
            showlegend=True,
            margin=go.layout.Margin(
                l=150,
                r=10,
                t=10,
            )
        )
    )

    # Convert the figures to JSON
    return json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

