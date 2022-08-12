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

# Import figure functions
import sys
import os


def load_ann(record_id, segment_id):
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

    filename = 'p{:05d}_s{:02d}'.format(record_id, segment_id)
    pn_dir = 'icentia11k-continuous-ecg/1.0/p{:02d}/p{:05d}/'.format(
        record_id//1000, record_id)

    try:
        ann = wfdb.rdann(filename, "atr", pn_dir=pn_dir)
        
    except:
        print('File not found for record_id={}, segment={}'.format(record_id,segment_id))
        return

    df_beats = pd.DataFrame({'sample': ann.sample,
                            'type': ann.symbol,
                            'rhythm': ann.aux_note}
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

    filename = 'p{:05d}_s{:02d}'.format(record_id,segment_id)
    pn_dir = 'icentia11k-continuous-ecg/1.0/p{:02d}/p{:05d}/'.format(
        record_id//1000, record_id)

    try:
        signals, fields = wfdb.rdsamp(filename,
                                      pn_dir=pn_dir,
                                      sampfrom=sampfrom,
                                      sampto=sampto,
                          )
    except:
        print('File not found for record_id={}, segment_id={}'.format(record_id,segment_id))
        df_ecg = pd.DataFrame({'sample':[], 'signal':[]})
        return df_ecg

    df_ecg = pd.DataFrame({'sample':np.arange(sampfrom, sampto),
                           'signal':signals[:,0]})

    return df_ecg





# import time
# start = time.time()
# df_ecg = load_ecg(0, 0, 0, 250*60*1)
# # df_ecg = pd.DataFrame({'sample':[], 'signal':[]})
# fig = make_ecg_plot(df_ecg, include_annotation=True)
# fig.write_html('temp2.html')
# end = time.time()
# print(end-start)



def get_nib_values(list_labels, verbose=0):
    '''
    Compute the NIB values for list of beat annotations list_labels

    A value of -1 means that NIB could not be computed due to interuption from
    noise values.

    Parameters:
        list_labels: list of 'N','V','Q','S' corresponding to beats

    '''
    nib = np.nan
    count = 0
    list_nib = []

    for idx, label in enumerate(list_labels):

        if label=='N':
            nib+=1
            count+=1

        if label=='V':

            # Convert Nan to -1 to keep all values integers
            nib = -1 if np.isnan(nib) else nib
            list_nib.extend([nib]*(count+1))

            # Reset counts
            nib=0
            count=0

        if label in ['Q', 'S']:
            nib=np.nan
            count+=1

        if verbose:
            if idx%10000==0:
                print('Complete for index {}'.format(idx))

    # Add the final labels (must be -1 if remaining)
    l_remaining = len(list_labels)-len(list_nib)
    if l_remaining > 0:
        list_nib.extend([-1]*l_remaining)

    return list_nib



def compute_rr_nib_nnavg(df_beats):
    '''
    Compute RR intervals, NIB values, and NN avg
    Places into input dataframe and returns

    Parameters
    ----------
    df_beats : pd.DataFrame
        Beat annotation labels.
        Cols ['sample', 'type']

    Returns
    -------
    df_beats : pd.DataFrame

    '''

    #------------
    # Compute RR intervals
    #--------------

    # Remove + annotation which indicates rhythm change
    df_beats = df_beats[df_beats['type'].isin(['N','V','S','Q'])].copy()

    # Compute RR intervals and RR type (NN, NV etc.)
    df_beats['interval'] = df_beats['sample'].diff()
    df_beats['type_previous'] = df_beats['type'].shift(1)
    df_beats['interval_type'] = df_beats['type_previous'] + df_beats['type']
    df_beats.drop('type_previous', axis=1, inplace=True)


    #------------
    # Compute NN avg over 1 minute intervals
    # Approximate by the average of all intervals of type NN, NV, VN
    #----------

    df_beats['minute'] = (df_beats['sample']//(250*60)).astype(int)

    df_temp = df_beats[df_beats['interval_type'].isin(['NN','NV','VN'])].copy()
    # Remove rows that have interval > 2s -  these are due to missing
    # noise label in data.
    anomalies = df_temp[df_temp['interval']>2*250].index
    df_temp = df_temp.drop(anomalies)
    nn_avg = df_temp.groupby('minute')['interval'].median()
    nn_avg.name = 'nn_avg'
    nn_avg = nn_avg.reset_index()
    df_beats = df_beats.merge(nn_avg, on='minute')


    #-----------
    # Compute NIB values
    #-----------

    list_nib = get_nib_values(df_beats['type'], verbose=0)
    df_beats['nib'] = list_nib

    return df_beats



def make_rr_plot(df_beats):

    cols = px.colors.qualitative.Plotly
    color_discrete_map = {'NN':cols[0], 'NV':cols[1],
                          'VN':cols[2], 'VV':cols[3],
                          'NN avg':cols[4]}

    df_beats['Time (min)'] = df_beats['sample']/250/60
    df_beats['interval'] = df_beats['interval']/250
    df_beats['nn_avg'] = df_beats['nn_avg']/250

    df_beats = df_beats[df_beats['interval_type'].isin(['NN','NV','VN','VV'])]

    df_intervals = df_beats[['Time (min)','interval','interval_type']]

    df_nnavg = df_beats[['Time (min)']].copy()
    df_nnavg['interval'] = df_beats['nn_avg']
    df_nnavg['interval_type'] = 'NN avg'

    df_plot = pd.concat([df_intervals, df_nnavg])

    fig = px.scatter(df_plot, x='Time (min)', y='interval',
                     color='interval_type',
                     color_discrete_map=color_discrete_map)

    fig.update_yaxes(title='Interval (s)')
    fig.update_layout(margin={'l':80,'r':150,'t':40,'b':30},
                      height=350,
                      legend_title_text='Interval type',
                      )
    return fig



def make_ecg_plot(df_ecg, include_annotation=False):

    df_ecg['Time (min)'] = df_ecg['sample']/250/60

    fig = px.line(df_ecg, x='Time (min)', y='signal')

    if include_annotation:
        fig.add_annotation(x=0.5, y=0.5, xref='x domain', yref='y domain',
                    text='ECG shows for a selected time window of less than 1 minute',
                    font=dict(size=20),
                    showarrow=False,
                    )
    fig.update_yaxes(title='Voltage (mV)')
    fig.update_layout(
            margin={'l':80,'r':150,'t':40,'b':30},
            height=350,
            # title=title,
            titlefont={'family':'HelveticaNeue','size':18},
            )

    return fig






#-------------
# Launch the dash app
#---------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

if os.path.isdir('/Users/tbury'):
    run_cloud=False
else:
    run_cloud=True

if run_cloud:
    requests_pathname_prefix = '/app-holter-icentia11k/'
else:
    requests_pathname_prefix = '/'

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                requests_pathname_prefix=requests_pathname_prefix
                )

server = app.server
print('Launching dash')




# Default record ID
record_id_def = 0
segment_id_def = 0

# Import default data
df_ann = load_ann(record_id_def, 0)
df_rr = compute_rr_nib_nnavg(df_ann)
df_ecg = pd.DataFrame({'sample':[], 'signal':[]})


# Make figures
fig_rr = make_rr_plot(df_rr)
fig_ecg = make_ecg_plot(df_ecg, include_annotation=True)


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



        # Loading animation
        html.Div(
            [
            dcc.Loading(
                id="loading-anim",
                type="default",
                children=html.Div(id="loading-output"),
#                 color='#2ca02c',
            ),
            ],
            style={'width':'10%',
                'height':'40px',
                # 'fontSize':'12px',
                'padding-left':'0%',
                'padding-right':'0%',
                'padding-bottom':'10px',
                'padding-top':'20px',
                'vertical-align': 'middle',
                'display':'inline-block',
                },

        ),


        # Interval plot layout
        html.Div(
            [
                dcc.Graph(id='rr_plot',
                       figure = fig_rr,
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
                dcc.Graph(id='ecg_plot',
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
                'Created and maintained by ',
            html.A('Thomas Bury',
                   href='http://thomas-bury.research.mcgill.ca/',
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
    [
     Output('rr_plot','figure'),
     # Output('ecg_plot','figure'),
      Output('loading-output','children'),
      ],
    [
      Input('dropdown_record_id','value'),
      Input('dropdown_segment_id','value'),
      ],
)

def modify_record(record_id_mod, segment_id_mod):
    '''
    Update dataframes based on change in dropdown box and slider

    '''

    # If dropdown box was cleared, don't do anything
    if (record_id_mod is None) or (segment_id_mod is None):
        raise dash.exceptions.PreventUpdate()

    # Import new data
    df_ann = load_ann(record_id_mod, segment_id_mod)
    df_rr = compute_rr_nib_nnavg(df_ann)

    fig_rr = make_rr_plot(df_rr)

    return [fig_rr,
            '',
            ]



@app.callback(
        [
        Output('ecg_plot','figure'),
          ],
        [
        Input('rr_plot','relayoutData'),
        Input('dropdown_record_id','value'), # we need these to import correct ECG data
        Input('dropdown_segment_id','value')
        ],
)

def modify_time_window(relayout_data, record_id, segment_id):

    # ctx provides info on which input was triggered
    ctx = dash.callback_context

    # If layout_data was triggered
    # print (ctx.triggered[0])
    if ctx.triggered[0]['prop_id'] == 'rr_plot.relayoutData':



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

    return [fig_ecg]



#-----------------
# Add the server clause
#–-----------------

if run_cloud:
    host='206.12.98.131'
else:
    host='127.0.0.1'

if __name__ == '__main__':
    app.run_server(debug=True,
                   host=host,
                   )
