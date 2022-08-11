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


import plotly.express as px

import wfdb

# Import figure functions
import sys
import os



record_id = 24
segment_id = 3


def load_ann(record_id, segment_id):
    
    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir = 'icentia11k-continuous-ecg/1.0/p{:02d}/p{:05d}/'.format(
        record_id//1000, record_id)

    ann = wfdb.rdann(filename, "atr", pn_dir=pn_dir)

    df_beats = pd.DataFrame({'sample': ann.sample,
                             'type': ann.symbol,
                             'rhythm': ann.aux_note}
                            )
    return df_beats



def load_ecg(record_id, segment_id, sampfrom, sampto):

    filename = 'p{:05d}_s{:02d}'.format(record_id,segment_id)
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



def make_ecg_plot(df_ecg):
    
    # Make a column for time elapsed in minutes
    df_ecg['Time (min)'] = df_ecg['sample']/250/60
    
    fig = px.line(df_ecg, x='Time (min)', y='signal')
    fig.update_yaxes(title='Voltage (mV)')
    fig.update_layout(
            margin={'l':80,'r':150,'t':40,'b':30},
            height=350,
            titlefont={'family':'HelveticaNeue','size':18},
            )

    return fig






df = load_ecg(1234, 1, 0, 1000)
df.set_index('sample').plot()












