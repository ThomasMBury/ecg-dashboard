#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:27:11 2022

Code snippets to go into Medium article

@author: tbury
"""



import numpy as np
import pandas as pd

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import matplotlib.pyplot as plt


import plotly.express as px

import wfdb

# Import figure functions
import sys
import os



record_id = 0
segment_id = 0
sampto=0


def load_beats(record_id, segment_id, sampfrom, sampto):
    
    # name and path to data stored on Physionet
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
    
    # name and path to data stored on Physionet
    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir_root = 'icentia11k-continuous-ecg/1.0/'
    pn_dir = 'p{:02d}/p{:05d}/'.format(record_id//1000, record_id)
    signals, fields = wfdb.rdsamp(filename,
                                  pn_dir=pn_dir_root+pn_dir,
                                  sampfrom=sampfrom,
                                  sampto=sampto
                                  )
    
    df_ecg = pd.DataFrame(
        {'sample': np.arange(sampfrom, sampfrom+len(signals)),
         'signal': signals[:,0]}
        )
    
    return df_ecg




def make_ecg_plot(df_ecg):
    
    # Make a column for time in minutes
    df_ecg['Time (min)'] = df_ecg['sample']/250/60
    
    fig = px.line(df_ecg,
                  x='Time (min)', 
                  y='signal',
                  labels={'signal':'Voltage (mV)'},
                  height=350)
    
    return fig




def make_beat_interval_plot(df_beats):

    # Make a column for time in minutes
    df_beats['Time (min)'] = df_beats['sample']/250/60
    
    # Make column for time interval between beats
    df_beats['Interval (s)'] = (df_beats['sample'] 
                                - df_beats['sample'].shift(1))
    
    # Make column for type of interval
    df_beats['Interval Type'] = (df_beats['type'].shift(1)
                                 + df_beats['type'])
    
    # Only consider intervals between N and V beats (NN, NV, VN or VV)
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
                     height=350
                     )    
    
    return fig





# Good section to use for figs in text
record_id = 18
segment_id = 0
sampfrom = 2*250*60
sampto = 3*250*60

df_beats = load_beats(record_id, segment_id, sampfrom, sampto)
df_ecg = load_ecg(record_id, segment_id, sampfrom, sampto)


fig = make_beat_interval_plot(df_beats)
fig.write_html('temp1.html')

fig = make_ecg_plot(df_ecg)
fig.write_html('temp2.html')





# #-------------
# # 'Wall of ink' ECG figure
# #-------------
# df_ecg = load_ecg(0, 0, 0, 1*60*60*250)
# df_ecg['Time (hr)'] = df_ecg['sample']/250/60/60
# df_ecg.set_index('Time (hr)', inplace=True)
# df_ecg['signal'].plot()
# plt.ylabel('Voltage (mV)')






