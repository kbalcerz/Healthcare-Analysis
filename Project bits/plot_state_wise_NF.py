import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from google.cloud import storage
import seaborn as sns
import os
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=False)


def getStateWiseSNF():
    return df.groupby(['State'], as_index = False)['Provider ID'].count().rename(columns = {'Provider ID' : 'Count'})
    

df = pd.read_csv('Nursing_2014_processed.csv')

state_count_df = getStateWiseSNF()

print(state_count_df.head())

state_count_df['text'] = state_count_df['State'] + '<br>'

data = [ dict(
        type='choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        autocolorscale = False,
        locations = state_count_df['State'],
        z = state_count_df['Count'].astype(float),
        locationmode = 'USA-states',
        text = state_count_df['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2                
            )
        ),
        colorbar = dict(
            title = "Number of Nursing Facilities"
        )
    ) ]

layout = dict(
        title = 'Number of Nursing Facilities in 2014 in each State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict( data=data, layout=layout )

py.iplot( fig, filename='snf-cloropleth-map' )
