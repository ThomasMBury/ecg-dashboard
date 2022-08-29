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

from dash import Dash, html, dcc, exceptions, callback_context
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
    fig.update_yaxes(fixedrange=True)
    
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
    html.Div(
        html.H4('ECG dashboard'),
        style={'width':'200px',
               'height':'60px',
               'padding-left':'2%',
               'display':'inline-block',
               }),

    html.Div([
        html.Label('Record ID'),
        dcc.Dropdown(id='dropdown_record_id',
                     value=record_id_def,
                     options=np.arange(11000))],
        style={'width':'20%',
               'height':'60px',
               'padding-left':'2%',
               'display':'inline-block',
               }),        

    html.Div([
        html.Label('Segment ID'),
        dcc.Dropdown(id='dropdown_segment_id',
                     value=segment_id_def,
                     options=np.arange(50))],
        style={'width':'20%',
               'height':'60px',
               'padding-left':'2%',
               'display':'inline-block',
               }),     
    
    html.Div(
        dcc.Graph(id='fig_intervals', figure=fig_intervals),
        ),
    
    html.Div(
        dcc.Graph(id='fig_ecg', figure=fig_ecg),    
        )
])




@app.callback(
     Output('fig_intervals','figure'),
     Input('dropdown_record_id','value'),
     Input('dropdown_segment_id','value')
)
def update_record(record_id, segment_id):
    
    # If dropdown box was cleared, don't do anything
    if (record_id is None) or (segment_id is None):
        raise exceptions.PreventUpdate()
    
    df_beats = load_beats(record_id, segment_id, 0, None)
    fig_intervals = make_beat_interval_plot(df_beats)
    
    return fig_intervals



@app.callback(
    Output('fig_ecg','figure'),
    Input('fig_intervals','relayoutData'),
    Input('dropdown_record_id','value'),
    Input('dropdown_segment_id','value'),
    )

def update_ecg_plot(relayout_data, record_id, segment_id):

    # callback_context provides info on which input was triggered
    ctx = callback_context

    # If layout_data was triggered
    if ctx.triggered[0]['prop_id'] == 'fig_intervals.relayoutData':

        if relayout_data==None:
            relayout_data={}

        # If neither bound has changed (other button may have been clicked)
        # then prevent update
        if ('xaxis.range[0]' not in relayout_data) and \
           ('xaxis.range[1]' not in relayout_data) and \
           ('xaxis.autorange' not in relayout_data):
            raise exceptions.PreventUpdate()

        # If auto-range button has been clicked
        if 'xaxis.autorange' in relayout_data:
            tmin_adjust = 0
            tmax_adjust = 60

        # If lower bound has been updated
        if ('xaxis.range[0]' in relayout_data):
            tmin_adjust = relayout_data['xaxis.range[0]']
        else:
            tmin_adjust = 0

        # If upper bound has been updated
        if ('xaxis.range[1]' in relayout_data):
            tmax_adjust = relayout_data['xaxis.range[1]']
        else:
            tmax_adjust = 60

    # If record_id or segment_id were triggered, reset ECG
    else:
        tmin_adjust = 0
        tmax_adjust = 60

    # If time window < 1 min, then import ECG data
    if tmax_adjust - tmin_adjust < 1:

        sampfrom = int(tmin_adjust*60*250)
        sampto = int(tmax_adjust*60*250)
        df_ecg = load_ecg(record_id, segment_id, sampfrom, sampto)
        
        # Make ECG figure
        fig_ecg = make_ecg_plot(df_ecg)

    else:
        # Make empty ECG figure
        df_ecg = pd.DataFrame({'sample':[], 'signal':[]})
        fig_ecg = make_ecg_plot(df_ecg)

    return fig_ecg






#################################

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    
    
    