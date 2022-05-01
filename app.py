#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1 May 2022

Dashboard to view ECG recording. Includes:
	Inter-beat interval time series (RR plot)
	ECG trace, which shows when user selects a sufficiently small 
        portion of RR plot
    Heartprint
    Phase response plots

To be deployed on Heroku 

@author: tbury

"""

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_auth

# Import figure functions
import sys
import os
import funs_figures as figs



def filter_time_range(df, trange):
    '''
    Filter dataframe df to be within trange.

    Parameters
    ----------
    df : DataFrame
        dataframe with at least the column 'Time (s)'
    trange: list
        list of min and max time (in hours)

    Returns
    -------
    Filtered dataframe

    '''
    
    # Convert trange to seconds
    tmin,tmax = np.array(trange)*60*60
    
    # Filter data to be within time range
    df_out = df[
        (df['Time (s)']>=tmin)&\
        (df['Time (s)']<=tmax)].copy()
    
    return df_out


def load_record_data(record_id):
    '''
    Load in record data from the cloud.
    Best to load in all data whilst app is initalising
    as opposed to loading within callback functions as this requires
    saving the data in the users browser which can slow app down.
    Loading everything upon deployment means the app takes time to initalise,
    however operations within the app are fast once it is going.
    
    Parameters
    ----------
    record_id : string
        ID of Holter recording

    Returns
    -------
    df_rr, df_nnavg, df_vv_stats

    '''
    
    # Read in pandas dataframes for beat data
    df_rr = pd.read_csv(
        fileroot+'rr_intervals/{}_rr.csv'.format(record_id))
    df_nnavg = pd.read_csv(
        fileroot+'nn_avg/{}_nnavg_rw4.csv'.format(record_id)) 
    df_vv_stats = pd.read_csv(
        fileroot+'vv_stats/{}_vv_stats.csv'.format(record_id)) 
    df_vv_stats['Time (s)'] = df_vv_stats['Time start (s)']
    
    return df_rr, df_nnavg, df_vv_stats


def get_record_specific_data(record_id, trange):
    '''
    Get record data for specific Holter and specific time range (in hours)
        
    Input:
        record_id: ID for patient
        trange: [tmin,tmax]
        
    Output:
        df_rr, df_nnavg, df_vv_stats
    '''   
    df_rr = filter_time_range(dic_df_rr[record_id], trange)
    df_nnavg = filter_time_range(dic_df_nnavg[record_id], trange)
    df_vv_stats = filter_time_range(dic_df_vv_stats[record_id], trange)   
    # df_vv_stats['Time (s)'] = df_vv_stats['Time start (s)']
    
    ## Merge NN avg data 
    df_rr = df_rr.merge(df_nnavg, on='Time (s)')
    df_vv_stats = df_vv_stats.merge(df_nnavg, on='Time (s)')
  
    return df_rr, df_nnavg, df_vv_stats




#--------------
# Import record data
#–-------------


# Fileroot on local computer
fileroot_local='/Users/tbury/Google Drive/Research/postdoc_21_22/ecg-dashboard/data/'

# Fileroot on Heroku (check)
fileroot_cloud='/home/ubuntu/holter_data/ubc_registry/data_output/'


# Work out which one to use
if os.path.isdir(fileroot_local):
	fileroot = fileroot_local
	# Boolean to determine if running on cloud or locally
	run_cloud = False
else:
	fileroot = fileroot_cloud
	run_cloud = True

print('Using data file root {}'.format(fileroot))


# list of ID for each patient to include in app
list_record_id = ['AK6890']

# Initialise dictionaries to store data from all records
dic_df_rr = {}
dic_df_nnavg = {}
dic_df_vv_stats = {}


# Import data for each record
print('Begin importing patient data to app')

# Loop throuh each record ID for this given patient
for record_id in list_record_id:
    
    # Check existence of output data
    if not os.path.exists(fileroot+'rr_intervals/{}_rr.csv'.format(record_id)):
        continue
    
    # Get beat data specific to this record
    df_rr, df_nnavg, df_vv_stats = load_record_data(record_id)       
    
    # Add data to dictionary
    dic_df_rr[record_id] = df_rr
    dic_df_nnavg[record_id] = df_nnavg
    dic_df_vv_stats[record_id] = df_vv_stats  
    
print('Finished importing patient data')


## Get markdown (md) file containing app description
# Filepath on local computer
desc_filepath_local='/Users/tbury/Google Drive/Research/postdoc_21_22/ecg-dashboard/'
# Fileroot on cloud
desc_filepath_cloud='/home/ubuntu/cloud_holter_app/app-holter/'
# Work out which one to use
if os.path.isdir(fileroot_local):
	desc_filepath = desc_filepath_local
else:
	desc_filepath = desc_filepath_cloud
f = open(desc_filepath+'description.md', 'r') 
description_text = f.read()
f.close()




#–----------
# Default settings (executed on initial run)
#–----------

# Default record ID
record_id_def = list_record_id[0]

# Initial time range of Holter recording
trange_def = [0,24]



#-------------
# Launch the dash app
#---------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print('Launching dash')

server = app.server

if run_cloud:
	app.config.update({  
	# Change this to directory name used in WSGIAlias
		'requests_pathname_prefix': '/app-holter-ubc-ecg/',
	})
	print('Updated prefix')



# Dropdown box settings (record_id options are updated in callback fn)
opts_record_id = [{'label':k,'value':k} for k in list_record_id]
# Tick marks on time range slider
time_marks = {int(x):str(x) for x in np.arange(0,180,24)}


#-----------------
# Create figures
#-----------------

# Set RR range
rr_range=[0.25,2.25]


# Static figure titles
title_rr = 'Beat-to-beat interval plot'
title_ecg = 'ECG trace (displays for time intervals < 0.1hr)'


# Get record specific data for default values
df_rr, df_nnavg, df_vv_stats = \
    get_record_specific_data(record_id_def,
                         trange_def,
                         )

# hookup_time = df_record_IDs[\
#    df_record_IDs['Record ID']==record_ID_def]['Start time (hr)'].values[0]

    
##----------- Interval plot--------------
fig_rr = figs.make_rr_plot_express(df_rr,
                                   rr_range=rr_range,
                                   title=title_rr,
                                   fixedrange=True,
)

#-----------ECG plot---------------
filepath_ecg = fileroot+'ecg_signals/{}_signal_data'.format(str(record_id))
fig_ecg = figs.make_ecg_plot(filepath_ecg, trange_def[0]*3600, trange_def[1]*3600,
                             title=title_ecg)



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

            html.H4('ECG dashboard analysis',
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
                         options=opts_record_id,
                         value=record_id_def,
                         optionHeight=20,
                         searchable=False,
                         clearable=False
            ),
            
            ],
                        
            style={'width':'20%',
                   'height':'60px',
                   'fontSize':'14px',
                   'padding-left':'0%',
                   'padding-right':'0%',
                   'padding-bottom':'10px',
                   'padding-top':'30px',
                   'vertical-align': 'middle',
                   'display':'inline-block'},       
        ),
           
        
        # Slider for time range
        html.Div(
            [
            # Slider for time range
            html.Label('Time range (hrs)',
                        id='trange_slider_text',
                        style={'fontSize':14},
            ),  
            
            dcc.RangeSlider(
                    id='trange_slider',
                    min=0,
                    max=180,
                    step=1,
                    marks= time_marks,
                    value=trange_def,
            ),
            ],
            
            style={'width':'25%',
               'height':'60px',
               'fontSize':'10px',
               'padding-left':'5%',
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
   
   
    
        
        # Text on tips for usage
        html.Div(
            [
            dcc.Markdown(description_text)
            ],
  
            style={'width':'30%',
                'height':'60px',
                'fontSize':'12px',
                'padding-left':'5%',
                'padding-right':'5%',
                'padding-bottom':'0px',
                'padding-top':'0px',
                # 'vertical-align': 'middle',
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
                   'height':'350px',
                   'fontSize':'10px',
                   'padding-left':'1%',
                   'padding-right':'1%',
                   'padding-top' : '40px',
                   'padding-bottom':'0px',
                   'vertical-align': 'middle',
                   'display':'inline-block'},
        ),
    
#       Loading animation 2
        html.Div(
            [       
            dcc.Loading(
                id="loading-anim2",
                type="default",
                children=html.Div(id="loading-output2")
            ),
            ],
            style={'width':'10%',
                'height':'40px',
                'fontSize':'12px',
                'padding-left':'90%',
                'padding-right':'0%',
                'padding-bottom':'0px',
                'padding-top':'0px',
                'vertical-align': 'middle',
                'display':'inline-block',
                },
                
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
                   'height':'350px',
                   'fontSize':'10px',
                   'padding-left':'1%',
                   'padding-right':'1%',
                   'padding-top' : '0px',
                   'padding-bottom':'20px',
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



# Update data upon change record ID or time range
@app.callback(
    [Output('rr_plot','figure'),
     Output('loading-output','children'), 
     ],
    [
     Input('dropdown_record_id','value'),
     Input('trange_slider','value'),
     ],
)


def update_patient(record_id_adjusted, trange_adjusted):
    '''
    Update dataframes based on change in dropdown box and slider
    
    '''
    # Load in new data
    df_rr_update, df_nnavg_update, df_vv_stats_update =\
        get_record_specific_data(record_id_adjusted, trange_adjusted)

    # Update interval plot
    fig_rr = figs.make_rr_plot_express(df_rr_update,
                                       rr_range=rr_range,
                                       title=title_rr,
                                       fixedrange=True,
    )
    
    return [fig_rr,
            # fig_ecg,
            # fig_24hour_stats,
            '',
            ]
    
    
# Update stats figures based on change in interval plot selection     
@app.callback(
        [Output('ecg_plot','figure'),
         Output('loading-output2','children'),
          ],
        [Input('rr_plot','relayoutData'),
         Input('dropdown_record_id','value'),
         Input('trange_slider','value'),
          ],
)

def update_stat_figures(layout_data, record_id, trange):
    
    #–--------Determine new values for tmin and tmax to display stats data-------#
    
    # ctx provides info on which input was triggered
    ctx = dash.callback_context
    
    # If layout_data was triggered
    # print (ctx.triggered[0])
    if ctx.triggered[0]['prop_id'] == 'rr_plot.relayoutData':
    
	    if layout_data==None:
		    layout_data={}
	
		# If neither bound has been changed (due to a click on other button) don't do anything
	    if ('xaxis.range[0]' not in layout_data) and ('xaxis.range[1]' not in layout_data) and ('xaxis.autorange' not in layout_data):
		    raise dash.exceptions.PreventUpdate()


		# If range has been auto-ranged
	    if 'xaxis.autorange' in layout_data:
		    tmin_adjust = trange[0]
		    tmax_adjust = trange[1]

		# If lower bound has been changed
	    if ('xaxis.range[0]' in layout_data):
		    # Adjusted lower bound
		    tmin_adjust = layout_data['xaxis.range[0]']
		    # If lower than lower bound, use tmin
		    tmin_adjust = max(trange[0],tmin_adjust)
	    else:
		    tmin_adjust = trange[0]
	   
		# If upper bound has been changed
	    if ('xaxis.range[1]' in layout_data):
			# Adjusted upper bound
		    tmax_adjust = layout_data['xaxis.range[1]']
			# If higher than higher bound, use tmax
		    tmax_adjust = min(trange[1],tmax_adjust)
	    else:
		    tmax_adjust = trange[1]
    
    # If recordID or trange was triggered
    else:
        tmin_adjust = trange[0]
        tmax_adjust = trange[1]
    
    
    # # Get adjusted stat data
    # df_rr_select, df_nnavg_select, df_vv_stats_select = \
    #     get_record_specific_data(record_id,np.array([tmin_adjust,tmax_adjust]))


    # Update ECG plot
    filepath_ecg = fileroot+'ecg_signals/{}_signal_data'.format(str(record_id))
    fig_ecg = figs.make_ecg_plot(filepath_ecg, tmin_adjust*3600, tmax_adjust*3600,
                                 title=title_ecg)
    
    # Return new plots
    return [fig_ecg,
            '',
            ]



#-----------------
# Add the server clause
#–-----------------

if run_cloud:
    host='206.12.93.144'
else:
    host='127.0.0.1'

if __name__ == '__main__':
    app.run_server(debug=True,
                   host=host,
                   )
    
    
    
