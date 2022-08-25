#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 27, 2022

Dash app to stream and visulaise the ECG and annotation data
from the Icentia11k database on Physionet located here
https://www.physionet.org/content/icentia11k-continuous-ecg/1.0/


@author: Thomas M. Bury

"""


import numpy as np
import pandas as pd

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.express as px

import wfdb




def load_beats(record_id, segment_id, sampfrom=0, sampto=None):
    '''
    Import segment of annotation labels from Icentia11k Physionet.

    Parameters
    ----------
    record_id : int between 0 and 10999
    segment_id : int between 0 and 49
        Note not all patients have 50 segments.

    Returns
    -------
    df_beats : pd.DataFrame
        Beat annotations

    '''
    
    # name and path to data stored on Physionet
    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir_root = 'icentia11k-continuous-ecg/1.0/'
    pn_dir = 'p{:02d}/p{:05d}/'.format(record_id//1000, record_id)

    try:
        ann = wfdb.rdann(filename, "atr", 
                         pn_dir=pn_dir_root+pn_dir,
                         sampfrom=sampfrom,
                         sampto=sampto
                         )        
    except:
        print('File not found for record_id={}, segment={}'.format(
            record_id,segment_id))
        return

    df_beats = pd.DataFrame({'sample': ann.sample,
                                 'type': ann.symbol}
                            )
    return df_beats



def load_ecg(record_id, segment_id, sampfrom, sampto):
    '''
    Import segment of ECG from Icentia11k Physionet.

    Parameters
    ----------
    record_id : int between 0 and 10999
    segment_id : int between 0 and 49
        Note not all patients have 50 segments.

    Returns
    -------
    df_ecg : pd.DataFrame

    '''

    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir_root = 'icentia11k-continuous-ecg/1.0/'
    pn_dir = 'p{:02d}/p{:05d}/'.format(record_id//1000, record_id)

    try:
        signals, fields = wfdb.rdsamp(filename,
                                      pn_dir=pn_dir_root+pn_dir,
                                      sampfrom=sampfrom,
                                      sampto=sampto
                                      )
    except:
        print('File not found for record_id={}, segment_id={}'.format(
            record_id,segment_id))
        df_ecg = pd.DataFrame({'sample':[], 'signal':[]})
        return df_ecg

    df_ecg = pd.DataFrame(
        {'sample': np.arange(sampfrom, sampfrom+len(signals)),
         'signal': signals[:,0]}
        )

    return df_ecg







def make_interval_plot(df_beats):

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
    fig.update_layout(margin={'l':80,'r':150,'t':40,'b':0})
    
    return fig


def make_ecg_plot(df_ecg, include_annotation=False):
    
    # Make a column for time in minutes
    df_ecg['Time (min)'] = df_ecg['sample']/250/60
    
 
    fig = px.line(df_ecg,
                  x='Time (min)', 
                  y='signal',
                  labels={'signal':'Voltage (mV)'},
                  height=300)
    
    if include_annotation:
        fig.add_annotation(x=0.5, y=0.5, xref='x domain', yref='y domain',
                    text='ECG shows for a selected time window of less than 1 minute',
                    font=dict(size=20),
                    showarrow=False,
                    )    
    fig.update_layout(margin={'l':80,'r':150,'t':40,'b':30})
    
    return fig




#-------------
# Launch the dash app
#---------------

app = dash.Dash(__name__,
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

server = app.server


# Default record ID
record_id_def = 0
segment_id_def = 0

# Import default data
df_beats = load_beats(record_id_def, 0)
df_ecg = pd.DataFrame({'sample':[], 'signal':[]})


# Make figures
fig_intervals = make_interval_plot(df_beats)
fig_ecg = make_ecg_plot(df_ecg)


#--------------------
# App layout
#–-------------------

# Font sizes
size_title = '20px'

app.layout = \
    html.Div([


        # Title section of app
        html.Div(
            [

            html.H4('Icentia11k Holter recordings',
                    style={
                        # 'textAlign':'center',
                        # 'fontSize':26,
                        'color':'black',
                        }
            ),
            ],

            style={'width':'30%',
                    'height':'60px',
                    'fontSize':'10px',
                    'padding-left':'2%',
                    'padding-right':'0%',
                    'padding-bottom':'10px',
                    'padding-top':'30px',
                    'vertical-align': 'middle',
                    'display':'inline-block'},

        ),

        # Dropdown menu for record_id
        html.Div(
            [

            # Label
            html.Label('Record ID',
                       style={'fontSize':14}
            ),

            dcc.Dropdown(id='dropdown_record_id',
                         options=np.arange(11000),
                         value=record_id_def,
                         optionHeight=20,
                         clearable=True,
            ),

            ],

            style={'width':'15%',
                   'height':'60px',
                   'fontSize':'14px',
                   'padding-left':'1%',
                   'padding-right':'0%',
                   'padding-bottom':'10px',
                   'padding-top':'30px',
                   'vertical-align': 'middle',
                   'display':'inline-block'},
        ),



        # Dropdown menu for segment_id
        html.Div(
            [

            # Label
            html.Label('Segment ID',
                       style={'fontSize':14}
            ),

            dcc.Dropdown(id='dropdown_segment_id',
                         options=np.arange(50),
                         value=segment_id_def,
                         optionHeight=20,
                         # searchable=False,
                          clearable=True
            ),

            ],

            style={'width':'15%',
                   'height':'60px',
                   'fontSize':'14px',
                   'padding-left':'1%',
                   'padding-right':'0%',
                   'padding-bottom':'10px',
                   'padding-top':'30px',
                   'vertical-align': 'middle',
                   'display':'inline-block'},
        ),



        # Interval plot layout
        html.Div(
            [
                dcc.Graph(id='fig_intervals',
                       figure = fig_intervals,
                       config={'doubleClick': 'autosize'}
                       )
            ],

            style={'width':'98%',
                   'height':'300px',
                   'fontSize':'10px',
                   'padding-left':'1%',
                   'padding-right':'1%',
                   'padding-top' : '40px',
                   'padding-bottom':'0px',
                   'vertical-align': 'middle',
                   'display':'inline-block'},
        ),


        # ECG plot layout
        html.Div(
            [
                dcc.Graph(id='fig_ecg',
                       figure = fig_ecg,
                       config={'doubleClick': 'autosize'}
                       )
            ],

            style={'width':'98%',
                   'height':'270px',
                   'fontSize':'10px',
                   'padding-left':'1%',
                   'padding-right':'1%',
                   'padding-top' : '40px',
                   'padding-bottom':'0px',
                   'vertical-align': 'middle',
                   'display':'inline-block'},
        ),


        # Footer
        html.Footer(
            [
                'Source code',
            html.A('here',
                   href='https://github.com/ThomasMBury/ecg-dashboard',
                   target="_blank",
                   ),
            ],
            style={'fontSize':'15px',
                              'width':'100%',
                               # 'horizontal-align':'middle',
                              'textAlign':'center',
                   },

            ),
])


#–-------------------
# Callback functions
#–--------------------


# Update data upon change of record_id or segment_id
@app.callback(
     Output('fig_intervals','figure'),
     Input('dropdown_record_id','value'),
     Input('dropdown_segment_id','value')
)

def modify_record(record_id_mod, segment_id_mod):
    '''
    Update dataframes based on change in dropdown box and slider

    '''

    # If dropdown box was cleared, don't do anything
    if (record_id_mod is None) or (segment_id_mod is None):
        raise dash.exceptions.PreventUpdate()

    # Import new data
    df_beats = load_beats(record_id_mod, segment_id_mod)
    fig_intervals = make_interval_plot(df_beats)

    return fig_intervals



@app.callback(
        Output('fig_ecg','figure'),
        Input('fig_intervals','relayoutData'),
        Input('dropdown_record_id','value'), # we need these to import correct ECG data
        Input('dropdown_segment_id','value'),
        )

def modify_time_window(relayout_data, record_id, segment_id):

    # ctx provides info on which input was triggered
    ctx = dash.callback_context

    # If layout_data was triggered
    # print (ctx.triggered[0])
    if ctx.triggered[0]['prop_id'] == 'fig_intervals.relayoutData':



        if relayout_data==None:
            relayout_data={}

        # If neither bound has been changed (due to a click on other button) don't do anything
        if ('xaxis.range[0]' not in relayout_data) and ('xaxis.range[1]' not in relayout_data) and ('xaxis.autorange' not in relayout_data):
            raise dash.exceptions.PreventUpdate()


        # If range has been auto-ranged
        if 'xaxis.autorange' in relayout_data:
            tmin_adjust = 0
            tmax_adjust = 60

        # If lower bound has been changed
        if ('xaxis.range[0]' in relayout_data):
            tmin_adjust = relayout_data['xaxis.range[0]']
        else:
            tmin_adjust = 0

        # If upper bound has been changed
        if ('xaxis.range[1]' in relayout_data):
            # Adjusted upper bound
            tmax_adjust = relayout_data['xaxis.range[1]']
        else:
            tmax_adjust = 60

    # If record_id or segment_id were triggered
    else:
        tmin_adjust = 0
        tmax_adjust = 60


    # If time window < 1 min, then import ECG data
    if tmax_adjust - tmin_adjust < 1:

        sampfrom = int(tmin_adjust*60*250)
        sampto = int(tmax_adjust*60*250)
        df_ecg = load_ecg(record_id, segment_id, sampfrom, sampto)
        # Make figure
        fig_ecg = make_ecg_plot(df_ecg)

    else:
        df_ecg = pd.DataFrame({'sample':[], 'signal':[]})
        fig_ecg = make_ecg_plot(df_ecg, include_annotation=True)

    return fig_ecg



#-----------------
# Add the server clause
#–-----------------


app.run_server(debug=True,
               host='127.0.0.1',
               )
    
    
    
