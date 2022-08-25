#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 27, 2022
@author: Thomas M. Bury

"""

import numpy as np
import pandas as pd
import plotly.express as px
import wfdb

from dash import Dash, html, dcc
from dash.dependencies import Input, Output

css = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = Dash(__name__, external_stylesheets=[css])
server = app.server


def load_beats(record_id, segment_id, sampfrom, sampto):
    
    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir_root = 'icentia11k-continuous-ecg/1.0/'
    pn_dir = 'p{:02d}/p{:05d}/'.format(record_id//1000, record_id)
    
    ann = wfdb.rdann(filename, "atr", 
                     pn_dir=pn_dir_root+pn_dir,
                     sampfrom=sampfrom,
                     sampto=sampto
                     )
    
    df_beats = pd.DataFrame({'sample': ann.sample,
                             'type': ann.symbol}
                            )
    return df_beats



def load_ecg(record_id, segment_id, sampfrom, sampto):

    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir_root = 'icentia11k-continuous-ecg/1.0/'
    pn_dir = 'p{:02d}/p{:05d}/'.format(record_id//1000, record_id)
    
    signals, fileds = wfdb.rdsamp(filename,
                                  pn_dir=pn_dir_root+pn_dir,
                                  sampfrom=sampfrom,
                                  sampto=sampto
                                  )
    
    df_ecg = pd.DataFrame({'sample': np.arange(sampfrom, sampto),
                           'signal': signals[:,0]})
    
    return df_ecg



def make_beat_interval_plot(df_beats):

    # Make a column for time in minutes
    df_beats['Time (min)'] = df_beats['sample']/250/60
    
    # Make column for time interval between beats
    df_beats['Interval (s)'] = (df_beats['sample'] 
                                - df_beats['sample'].shift(1))
    
    # Make column for type of interval
    df_beats['Interval Type'] = (df_beats['type'].shift(1)
                                 + df_beats['type'])
    
    # Only consider intervals between N and V beats (NN, NV, VN, VV)
    df_beats = df_beats[
        df_beats['Interval Type'].isin(['NN','NV','VN','VV'])]
    df_beats = df_beats.dropna()
    
    # Assign colours to each interval type
    cols = px.colors.qualitative.Plotly
    color_discrete_map = dict(zip(['NN','NV','VN','VV'], cols[:4]))
    fig = px.scatter(df_beats, 
                     x='Time (min)', 
                     y='Interval (s)',
                     color='Interval Type',
                     color_discrete_map=color_discrete_map,
                     height=300
                     )    
    
    fig.update_layout(margin={'l':80,'r':150,'t':40,'b':30})
    
    return fig


def make_ecg_plot(df_ecg):
    
    # Make a column for time in minutes
    df_ecg['Time (min)'] = df_ecg['sample']/250/60
    
    fig = px.line(df_ecg,
                  x='Time (min)', 
                  y='signal',
                  labels={'signal':'Voltage (mV)'},
                  height=300)
    
    if len(df_ecg)==0:
        fig.add_annotation(x=0.5, y=0.5, xref='x domain', yref='y domain',
                    text='Corresponding ECG for time windows < 1 minute',
                    font=dict(size=20),
                    showarrow=False,
                    )   
        
    fig.update_layout(margin={'l':80,'r':150,'t':40,'b':30})

    return fig


# Default patient and segment
record_id_def = 0
segment_id_def = 0

# Load defualt data
df_beats = load_beats(record_id_def, segment_id_def, 0, None)
df_ecg = pd.DataFrame({'sample':[], 'signal':[]})

# Make default figures
fig_intervals = make_beat_interval_plot(df_beats)
fig_ecg = make_ecg_plot(df_ecg)


#################################
# App details go in here

app.layout = html.Div([
    # List all components of app here
    html.H1('ECG Dashboard'),

    html.Div(
        dcc.Graph(id='fig_intervals', figure=fig_intervals),
        ),

    html.Div(
        dcc.Graph(id='fig_ecg', figure=fig_ecg), 
        )
])

#################################

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    
    
    