#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:20:22 2020

Functions to construct figures for visualising Holter recording

@author: tbury
"""


import numpy as np
import pandas as pd

import wfdb
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.express as px
import scipy.stats as stats


def make_fig_sinus_mod_scatter(df_rr,
                               diag_line=True,
                               title=''):
    '''
    Make scatter plot of (NV+VN)/2 vs NN (previous) to
    get a sense of how the ectopic beat modifies the sinus rhythm

    Parameters
    ----------
    df_rr : pd.DataFrame
        Columns ['Time (s)', RR interval (s), Type, Morphology]

    Returns
    -------
    plotly figure

    '''
    
    
    # Make shifted columns to collect all sequences of (NN, NV, VN)
    df_rr['Type+1'] = df_rr['Type'].shift(-1)
    df_rr['Type+2'] = df_rr['Type'].shift(-2)
    
    df_temp = df_rr[(df_rr['Type']=='NN') &\
              (df_rr['Type+1']=='NV') &\
              (df_rr['Type+2']=='VN')
              ].copy()
        
    # Assign RR interval lengths to each component
    df_temp['NN (s)'] = df_rr.iloc[df_temp.index]['RR interval (s)'].values
    df_temp['NV (s)'] = df_rr.iloc[df_temp.index+1]['RR interval (s)'].values
    df_temp['VN (s)'] = df_rr.iloc[df_temp.index+2]['RR interval (s)'].values
    
    # Make df with RR intervals for each sequence of (NN, NV, VN)
    df_triples = df_temp[['Time (s)','NN (s)','NV (s)','VN (s)']].copy()
    # Compute (NV+VN)/2
    df_triples['(NV+VN)/2 (s)'] = (df_triples['NV (s)'] + df_triples['VN (s)'])/2
    
    df_triples['Time (hr)'] = (df_triples['Time (s)']/3600).round(3)
    
    # # Make plotly figure
    # fig = px.scatter(df_triples,
    #                  x='NN (s)',
    #                  y='(NV+VN)/2 (s)',
    #                  hover_data=['Time (hr)'],
    #                  )
    
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(x=df_triples['NN (s)'],
                     y=df_triples['(NV+VN)/2 (s)'],
                     mode='markers',
                     marker={'size':2},
                     text=df_triples['Time (hr)'],
                     hovertemplate =
                        '<b>NN</b>: %{x:.3f}<br>'+
                        '<b>(NV+VN)/2</b>: %{y:.3f}<br>'+
                        '<b>Time (hr)</b>: %{text:.3f}<br>'+
                        '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor='purple',
                        font={'color':'white'}),                     
                    )
                 )
    
    # Add a trace for the line y=x
    xmin = df_triples['NN (s)'].min()
    xmax = df_triples['NN (s)'].max()
    # ymin = df_temp['(NV+VN)/2 (s)'].min()
    # ymax = df_temp['(NV+VN)/2 (s)'].max()
    if diag_line:
        fig.add_trace(
            go.Scatter(x=np.linspace(xmin,xmax,1000),
                       y=np.linspace(xmin,xmax,1000),
                       mode='lines',
                       line={'color':'black',
                             'dash':'dash'},                       
                       )
        )
    fig.update_layout(showlegend=False,
                      title=title)

    return fig
    
    
    





def make_fig_pvc_nn_nib(df_rr,
                       df_vv_stats,
                       nn_min = 0.5,
                       nn_max = 1.5,
                       nn_range_auto = True,
                       nn_interval_width = 0.008,
                       beat_threshold = 50,
                       title='PVC-NN relationship',
                       fig_height=300,
                       ):
    '''
    Make plot of %PVC vs NN with NIB colouring
    
    Input:
        df_rr: DataFrame with cols
            [Time (s), RR interval (s), Type, Morphology, NN avg]
        df_vv_stats: DataFrame with cols
            [Time (hr), NIB, VV, NV, VN, NN avg]
        nn_min: Min NN avg value to compute
        nn_max: Max NN avg value to compute
        nn_interval_width: Range of each histogram box
        beat_threshold: only include bins with #beats>beat_threshold
        
    Ouptut
        DataFrame with cols
            [NN avg, %PVC, NIB, Total beats]
    '''

    # Option to get nn_min and nn_max from data
    if nn_range_auto:
        nn_min = np.floor(df_rr['NN avg'].min()*100)/100
        nn_max = np.ceil(df_rr['NN avg'].max()*100)/100
        
    nn_vals = np.arange(nn_min, nn_max, nn_interval_width)
    
    list_tuples = []
    
    for nn_base in nn_vals: 
        # Upper value of bin
        nn_lid = nn_base + nn_interval_width
        # Select data with NN avg within bin
        df_rr_select = df_rr[(df_rr['NN avg']>=nn_base)&(df_rr['NN avg']<nn_lid)]
        df_vv_stats_select = df_vv_stats[(df_vv_stats['NN avg']>=nn_base)&(df_vv_stats['NN avg']<nn_lid)]
        
        # Compute total number of ectopic beats within this sinus range
        total_sinus = len(df_rr_select[df_rr_select['Type'].isin(['NN','VN','VN_inter'])])
        total_ectopic = len(df_rr_select[df_rr_select['Type'].isin(['NV','VV'])])
        total_beats = total_sinus + total_ectopic     
        
		# Compute proportion ectopic (if beats exist)
        if total_beats > 0:
            prop_ectopic = total_ectopic/total_beats
        else:
            prop_ectopic = np.nan
            
        # Compute proportion of data that has NIB 0,1,2,3,4, and >=5
        nib_prop = [len(df_vv_stats_select[df_vv_stats_select['NIB']==nib]) for nib in [0,1,2,3,4]]
        # Append proportion with NIB >=5
        nib_prop.append(len(df_vv_stats_select[df_vv_stats_select['NIB']>=5]))
        
        # Normalise list
        ar_nib_prop = np.array(nib_prop)
        # If no NIB values there could still be PVCs in data, but always seperated by noise.
        # In this case set to NIB>=5.
        ar_nib_prop = ar_nib_prop/sum(nib_prop) if sum(nib_prop)!=0 else np.array([0,0,0,0,0,1])
        # Put into dictionary
        dic_nib_prop = {'0':ar_nib_prop[0],
                        '1':ar_nib_prop[1],
                        '2':ar_nib_prop[2],
                        '3':ar_nib_prop[3],
                        '4':ar_nib_prop[4],
                        '>=5':ar_nib_prop[5]}
        
        for key in dic_nib_prop.keys():
            # %PVC contribution from this NIB category
            prop_ectopic_nib = dic_nib_prop[key] * prop_ectopic
            # Make a tuple with nn_avg (middle of bin), NIB value,
            # and %PVC contribution from this NIB category, and total beats
            tup = ((nn_base+nn_lid)/2, 
                   key,
                   prop_ectopic_nib,
                   total_beats)
            
            # Add tuple to list
            list_tuples.append(tup)            

        
    # Put into a dataframe
    dic_pvc_vs_nn = {'NN avg':[tup[0] for tup in list_tuples],
                     'NIB':[tup[1] for tup in list_tuples],
                     '%PVC':[tup[2] for tup in list_tuples],
                     'Total beats':[tup[3] for tup in list_tuples],
                     }
    df_pvc_nn = pd.DataFrame(dic_pvc_vs_nn)


    # Only keep data with total beats over beat_threshold
    df_pvc_nn_filt = df_pvc_nn[df_pvc_nn['Total beats']>beat_threshold]


    # Create figure
    if len(df_pvc_nn_filt)>0:
        # If data, make plot
        fig = px.bar(df_pvc_nn_filt,
                      x='NN avg',
                      y='%PVC',
                      color='NIB',
                      hover_data=['Total beats']
                      )
    else:
        # Blank figure (no data)
        fig = px.bar(df_pvc_nn_filt,
                     x='NN avg',
                     y='%PVC'
                     )
    
    fig.update_layout(bargap=0,
                      xaxis={'range':[nn_min,nn_max]},
                      title=title,
                      height=fig_height,
                      )
        
    return fig





def make_hist_morphology(df_rr,
                         xrange=[0.4,0.7],
                         title='',
                         bin_width = 0.008,
                         fig_height=300,
                         ):
    '''
    Make histogram of coupling interval coloured by morphology of PVC

    Parameters
    ----------
    df_rr : pd.DataFrame
        Interval data. Frame has cols 
        ['Time (s)','RR interval (s),Type, Morphology]
    xrange : list, optional
        x-axis range. The default is [0.4,0.7].
    title : string, optional
        Title of plot. The default is ''.
    bin_width : float, optional
        Width of histogram bins. Make a multiple of 0.004 (the precision
        of measurements). The default is 0.008.

    Returns
    -------
    fig : plotly fig object

    '''    
    
    # Select all coupling interval data
    df_nv = df_rr[df_rr['Type']=='NV']
    
    # Rename cols for plot
    df_nv = df_nv.rename(columns={'Time (s)':'Time (s)',
                  'RR interval (s)':'NV (s)',
                  'Type':'Type',
                  'Morphology':'Morphology'})
    
    # Generate figure
    fig = px.histogram(df_nv,
                       x='NV (s)',
                       color='Morphology',
    )
    
    # Update labels
    fig.update_layout(
        xaxis={
            'range': xrange,
            'title':'NV  (s)',
        },
        yaxis={'title':'Count'},
        title=title,
        barmode='overlay',
        bargap=0,
        height=fig_height,
    )

    # Reduce opacity to see both histograms
    fig.update_traces(
        opacity=0.75,
        xbins={'start':xrange[0],'end':xrange[1],'size':bin_width},
    )

    return fig



#----------------
# RR interval plot using Plotly graph_objects
#–---------------

def make_rr_plot(df_rr):
    '''
    Dot plot of intervals of the Holter recording over time
    
    Input:
        df_rr: dataframe of interval data
        	with cols ['Time (s)', RR interval (s)', 'Type', ''Morphology']
        	
    '''

	# Add a column for time in hours
    df_rr['Time (hr)'] = df_rr['Time (s)']/3600

    # Figure parameters
    height_intervals=200
    
    # Generate figure
    fig = go.Figure()


    # Add trace of NN beats
    fig.add_trace(go.Scatter(
            x=df_rr[df_rr['Type']=='NN']['Time (hr)'],
            y=df_rr[df_rr['Type']=='NN']['RR interval (s)'],
            name = 'NN',
            mode='markers',
            marker_color='blue'
            ))
    
    # Add trace of NV beats
    fig.add_trace(go.Scatter(
            x=df_rr[df_rr['Type']=='NV']['Time (hr)'],
            y=df_rr[df_rr['Type']=='NV']['RR interval (s)'],
            name = 'NV',
            mode='markers',
            marker_color='red'
            ))


    # Add trace of VN beats
    fig.add_trace(go.Scatter(
            x=df_rr[df_rr['Type']=='VN']['Time (hr)'],
            y=df_rr[df_rr['Type']=='VN']['RR interval (s)'],
            name = 'VN',
            mode='markers',
            marker_color='green'
            ))

    # Add trace of VV beats
    fig.add_trace(go.Scatter(
            x=df_rr[df_rr['Type']=='VV']['Time (hr)'],
            y=df_rr[df_rr['Type']=='VV']['RR interval (s)'],
            name = 'VV',
            mode='markers',
            marker_color='purple'
            ))    
    
    # Inteval axes
    fig.update_yaxes(
    				 range=[0.2,1.75],
    				 fixedrange=True,
                     title='Interval (s)')
    fig.update_xaxes(title='Time (hr)')    
    
    fig.update_layout(
            autosize=True,
            margin={'l':80,'r':150,'t':40,'b':0},
            height=height_intervals)
    
    
    return fig





def make_rr_plot_express(df_rr, 
                    autosize=False, 
                    fixedrange=False,
                    title='Beat-to-beat interval plot',
                    rr_range=[0.2,1.75]):
    '''
    Dot plot of RR intervals over time using Plotly express
    Appears to be faster than using Plotly graph objects
    
    Input:
        df_rr: dataframe of interval data 
            with cols [Time (s), RR interval (s), Type, Morphology, NN avg]
            
        autosize: Whether of not to fit the figure to the screen
        fixedYrange: whether or not to fix the y-axis range
            during interaction,
        rr_range: plot range of RR interval. If False, then
            get from data.
    Output:
        plotly figure

    '''
    
    # Make NN avg a Type in df_rr and move value to RR interval (s)
    df_nnavg = df_rr[['Time (s)','NN avg']].copy()
    df_nnavg.rename(columns={'NN avg':'RR interval (s)'},inplace=True)
    df_nnavg['Type'] = 'NN avg'
    df_nnavg['Morphology'] = 0
    df_plot = pd.concat([df_rr,df_nnavg],axis=0)
    
    
	# Add a column for time in hours
    df_plot['Time (hr)'] = df_plot['Time (s)']/3600    
    # Remove col for time in seconds
    df_plot = df_plot[['Time (hr)','RR interval (s)','Type','Morphology']].copy()
    
    # Remove noise entries
    df_plot.drop(df_plot[df_plot['Type']=='O'].index,inplace=True)
    
    # Figure params
    tmin = df_plot.iloc[0]['Time (hr)']
    tmax = df_plot.iloc[-1]['Time (hr)']
    
    # Get RR plot range if not provided
    if not rr_range:
        rr_min = df_plot['RR interval (s)'].min()
        rr_max = df_plot['RR interval (s)'].max()
        rr_range = [rr_min-0.1, rr_max+0.1] 
    
    
    # Add irrelevant points preceding df_rr to set color scheme
    df_plot.loc[-5] = [tmin,-1,'NN',0]
    df_plot.loc[-4] = [tmin,-1,'NV',0]
    df_plot.loc[-3] = [tmin,-1,'VN',0]
    df_plot.loc[-2] = [tmin,-1,'VV',0]
    df_plot.loc[-1] = [tmin,-1,'NN avg',0]
    
    df_plot.index = df_plot.index+5
    df_plot.sort_index(inplace=True)

    fig = px.scatter(df_plot, 
                     x='Time (hr)', 
                     y='RR interval (s)', 
                     color='Type',
                     hover_data=['Morphology'])


    # Inteval axes
    fig.update_yaxes(
    				 range=rr_range,
                     fixedrange=fixedrange,
                     title='Interval (s)')
    
    fig.update_xaxes(title='Time (hr)',
                     range=[tmin-0.5,tmax+0.5])    
    
    if not autosize:
        fig.update_layout(
                margin={'l':80,'r':150,'t':40,'b':0},
                height=350,
                title=title,
                titlefont={'family':'HelveticaNeue','size':18},
                )
    return fig








#----------------
# Make ECG plot
#----------------

def make_ecg_plot(filepath, tstart, tend, v_range=[-2,2], title=''):
    '''
    Make an ECG plot from a Physionet data file.

    Parameters
    ----------
    filepath : str
        path to Physionet files
    tstart : float
        start time of plot (s)
    tend : float
        end time of plot (s)
    vrange : interval of floats:
        range for y-axis (voltage)
    title : plot title

    Returns
    -------
    Plotly fig.

    '''

    # If path doesn't exist, don't try to create fig
    if not os.path.exists(filepath+'.dat'):
        fig = px.line(pd.DataFrame({'Time (hr)':[],'Voltage (mV)':[]}),
                      x='Time (hr)',
                      y='Voltage (mV)')
        fig.update_layout(
                margin={'l':80,'r':150,'t':40,'b':0},
                height=350,
                title=title,
                titlefont={'family':'HelveticaNeue','size':18},
                )   
        return fig
    
    
    # If tend-tstart is more than 0.1 hr, do not get ECG (takes too long)
    if tend-tstart > 360:
        fig = px.line(pd.DataFrame({'Time (hr)':[],'Voltage (mV)':[]}),
                      x='Time (hr)',
                      y='Voltage (mV)')
        fig.update_layout(
                margin={'l':80,'r':150,'t':40,'b':0},
                height=350,
                title=title,
                titlefont={'family':'HelveticaNeue','size':18},
                )      
        return fig


    # Convert tstart and tend into sample numbers
    sample_rate = 249.46785818085837 # Hz
    # sample_rate = 250 # Hz
    sample_start = int(np.floor(tstart*sample_rate))
    sample_end = int(np.floor(tend*sample_rate))
    

    # Get ECG wave data in form of wfdb object
    signals, fields = wfdb.rdsamp(filepath, 
                                  sampfrom=sample_start, 
                                  sampto=sample_end,
                                  )

    # Put into pandas df
    dict_ecg = {'Time (s)': np.arange(0,len(signals))*(1/sample_rate)+tstart}
    for j in range(signals.shape[1]):
        dict_ecg['channel{}'.format(j+1)] = signals[:,j]
    df_ecg = pd.DataFrame(dict_ecg)


    # Time in hours
    df_ecg['Time (hr)'] = df_ecg['Time (s)']/(3600)
    df_ecg.drop('Time (s)', inplace=True, axis=1)
    
    df_plot = df_ecg.melt(id_vars='Time (hr)', var_name='Channel', value_name='Voltage (mV)')
    
    fig = px.line(df_plot, x='Time (hr)', y='Voltage (mV)',
                  color='Channel')
    
    # Inteval axes
    fig.update_yaxes(range=v_range)
    
    fig.update_xaxes(title='Time (hr)',
                     range=[df_ecg['Time (hr)'].min(),
                            df_ecg['Time (hr)'].max()],
                            )
    
    fig.update_layout(
            margin={'l':80,'r':150,'t':40,'b':0},
            height=350,
            title=title,
            titlefont={'family':'HelveticaNeue','size':18},
            )    

    return fig
    








#–--------------
# Heartprint
#–---------------
def make_heartprint(df_nnavg,
                df_vv_stats,
                nib_range = [0,10],
                nn_range = [0.4,1.1],
                vv_range = [0,5],
                vn_range = [0.5,1.4],
                nv_range = [0.3,0.7],
                title='Heartprint',
                norm='total',
                bin_width = 0.008,
                ):
    
    '''
    Function to make a heartprint from data
    
    Input:
        df_nnavg: Dataframe of beats and average sinus rhythm.
            Has columns [Time (s), NN avg]
            
        df_vv_stats: Dataframe containing VV interval statistics. 
            Has columns [Time (s), NIB, VV, VN , NV, NN avg]
            
		'var'_range: Plotting range for the statistic 'var'.
        
        nn_range: If False, then work out from data
		
		title: Title of the heartprint
            
        norm: Method for normalising the frequency occurnece in the
            heatmaps. Options inlclude:
                'total': divide boxes by the total frequency occurence along a particular avg NN row
                'max': divide boxes by maximum frequency occurence along a particular avg NN row
                
        bin_width: Width of histogram bins (s)
            Take as a multiple of precision (0.004) to avoid aliasing
            
    Output:
        fig: Plotly figure of heartprint
    '''

    #-----------------
    # Fixed parameter values
    #–----------------
    
    height_histo = 400 # Height of histogram figures (fixed)
    
    bin_width = 0.008 # Width of each box of heatmap (s)
                    # NOTE: this should be a multiple of the recording precision
                    # which is 0.004 seconds (250Hz) to avoid aliasing.
    
    # Relative widths
    width_nib = 1.1
    width_vv = 1
    width_vn = 1
    width_nv = 1
    width_nn = 0.3
    x_spacing = 0.2 # Horizontal distance between heat maps
    y_spacing = 0.05 # Distance between heat maps and histograms
    
    # x-domain sizes
    tot_width = width_nib + width_vv + width_vn + width_nv + width_nn + 3*x_spacing + y_spacing
    domain_nib = np.array([0,width_nib/tot_width])
    domain_vv = domain_nib[1] + x_spacing/tot_width + np.array([0,width_vv])/tot_width
    domain_vn = domain_vv[1] + x_spacing/tot_width + np.array([0,width_vn])/tot_width
    domain_nv = domain_vn[1] + x_spacing/tot_width + np.array([0,width_nv])/tot_width
    domain_nn = domain_nv[1] + y_spacing/tot_width+ np.array([0,width_nn])/tot_width
    domain_nn[1]=1



    # NN range done automatically if nn_range=False
    if not nn_range:
        # Get min and max value of nn_avg
        nn_min = np.floor(100*df_vv_stats['NN avg'].min())/100
        nn_max = np.ceil(100*df_vv_stats['NN avg'].max())/100
        nn_range = [nn_min-0.1,nn_max+0.1]
        
        
    # If df_vv_stats only has one entry, histograms don't work
    # Therefore make it empty in this case
    if len(df_vv_stats)==1:
        df_vv_stats=pd.DataFrame([],columns=df_vv_stats.columns)

    
    #----------------
    # NIB traces
    #–---------------
    
    # Create normalised heatmap data
    nib_min,nib_max = [-0.5,20.5]
    nn_min,nn_max = [0.4,1.4]
    nib_edges = np.arange(nib_min,nib_max,1)  
    nn_edges = np.arange(nn_min, nn_max, bin_width)  
    x=df_vv_stats['NIB']
    y=df_vv_stats['NN avg']
    
    hist_data, nib_edges, nn_edges = np.histogram2d(x, y, bins=(nib_edges, nn_edges))
    hist_data = hist_data.T
    
    # Normalise each row (make sure that total occurence is non zero
    total_occurences = hist_data.sum(axis=1)
    max_occurences = hist_data.max(axis=1)
    # If normalising using total frequency
    if norm == 'total':
        hist_data_norm = np.array(
            [(hist_data[i]/total_occurences[i] if total_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])
    # If normalising using maximum frequency
    if norm == 'max':
        hist_data_norm = np.array(
            [(hist_data[i]/max_occurences[i] if max_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])
    
   
    # 2D histogram trace
    trace_hist2d_nib = go.Heatmap(
            x=nib_edges,
            y=nn_edges,
            z=hist_data_norm,
            xaxis='x1',
            yaxis='y1',
            colorscale='Blues',
            showscale=False,
            hovertemplate=
                '<b>NIB</b>: %{x:i}<br>'+
                '<b>NN</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}), 
    )
    

    # NIB histogram trace
    trace_nib = go.Histogram(
        x=df_vv_stats['NIB'], 
        marker=dict(color='gray'),
        xaxis='x1',
        yaxis='y2',
        xbins=dict(start=nib_min, end=nib_max, size=1),
        hovertemplate=
            '<b>NIB</b>: %{x:i}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),    
    )
    
    
    #–------------------
    # VV traces
    #–-------------------
    
    # Create normalised heatmap data
    vv_edges = np.arange(vv_range[0]-0.1, vv_range[1]+0.1,bin_width*4)  
    x=df_vv_stats['VV']
    y=df_vv_stats['NN avg']
    
    hist_data, vv_edges, nn_edges = np.histogram2d(x, y, bins=(vv_edges, nn_edges))
    hist_data = hist_data.T
    
    # Normalise each row (make sure that total occurence is non zero
    total_occurences = hist_data.sum(axis=1)
    max_occurences = hist_data.max(axis=1)
    # If normalising using total frequency
    if norm == 'total':
        hist_data_norm = np.array(
            [(hist_data[i]/total_occurences[i] if total_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])
    # If normalising using maximum frequency
    if norm == 'max':
        hist_data_norm = np.array(
            [(hist_data[i]/max_occurences[i] if max_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])    
    
    
    
    trace_hist2d_vv = go.Heatmap(
            x=vv_edges, 
            y=nn_edges, 
            z=hist_data_norm,
            name='density',
            xaxis='x2',
            yaxis='y3',
            colorscale='Blues',
            showscale=False,
            hovertemplate=
                '<b>VV</b>: %{x:.3f}<br>'+
                '<b>NN</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}),     
    )
    
    
    trace_vv = go.Histogram(
        x=df_vv_stats['VV'], 
        marker=dict(color='gray'),
        xbins=dict(start=vv_range[0], end=vv_range[1], size=bin_width*4),
        xaxis='x2',
        yaxis='y4',
        hovertemplate=
            '<b>VV</b>: %{x:.3f}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )
    
    
    
    
    
    #–----------------
    # VN traces
    #–----------------
        
    
    # Create normalised heatmap data
    vn_edges = np.arange(vn_range[0]-0.025, vn_range[1]+0.025,bin_width*2)  
    x=df_vv_stats['VN']
    y=df_vv_stats['NN avg']
    
    hist_data, vn_edges, nn_edges = np.histogram2d(x, y, bins=(vn_edges, nn_edges))
    hist_data = hist_data.T
    
    # Normalise each row (make sure that total occurence is non zero
    total_occurences = hist_data.sum(axis=1)
    max_occurences = hist_data.max(axis=1)
    # If normalising using total frequency
    if norm == 'total':
        hist_data_norm = np.array(
            [(hist_data[i]/total_occurences[i] if total_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])
    # If normalising using maximum frequency
    if norm == 'max':
        hist_data_norm = np.array(
            [(hist_data[i]/max_occurences[i] if max_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])   
        
    trace_hist2d_vn = go.Heatmap(
            x=vn_edges, 
            y=nn_edges, 
            z=hist_data_norm,
            xaxis='x3',
            yaxis='y5',
            colorscale='Blues',
            showscale=False,
            hovertemplate=
                '<b>VN</b>: %{x:.3f}<br>'+
                '<b>NN</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}),     )
    
    
    trace_vn = go.Histogram(
        x=df_vv_stats['VN'],
        marker=dict(color='gray'),
        xbins=dict(start=vn_range[0], end=vn_range[1], size=bin_width*2),
        xaxis='x3',
        yaxis='y6',
        hovertemplate=
            '<b>VN</b>: %{x:.3f}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )
    
    
    
    
    
    #----------------
    # NV plot
    #–---------------
    
    
    # Create normalised heatmap data
    nv_edges = np.arange(nv_range[0]-0.025, nv_range[1]+0.5,bin_width)  
    x=df_vv_stats['NV']
    y=df_vv_stats['NN avg']
    
    hist_data, nv_edges, nn_edges = np.histogram2d(x, y, bins=(nv_edges, nn_edges))
    hist_data = hist_data.T
    
    # Normalise each row (make sure that total occurence is non zero
    total_occurences = hist_data.sum(axis=1)
    max_occurences = hist_data.max(axis=1)
    # If normalising using total frequency
    if norm == 'total':
        hist_data_norm = np.array(
            [(hist_data[i]/total_occurences[i] if total_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])
    # If normalising using maximum frequency
    if norm == 'max':
        hist_data_norm = np.array(
            [(hist_data[i]/max_occurences[i] if max_occurences[i] else hist_data[i]) for i in range(len(hist_data))
             ])
 
      

      
    trace_hist2d_nv = go.Heatmap(
            x=nv_edges, 
            y=nn_edges,
            z=hist_data_norm,
            xaxis='x4',
            yaxis='y7',
            colorscale='Blues',
            hovertemplate=
                '<b>NV</b>: %{x:.3f}<br>'+
                '<b>NN</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}),     )
    
    
    trace_nv = go.Histogram(
        x=df_vv_stats['NV'],
        marker=dict(color='gray'),
        xaxis='x4',
        yaxis='y8',
        xbins=dict(start=nv_range[0], end=nv_range[1], size=bin_width),
        hovertemplate=
            '<b>NV</b>: %{x:.3f}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )
        
    trace_nn = go.Histogram(
        y=df_nnavg['NN avg'],
        marker=dict(color='gray'),
        ybins=dict(start=nn_min, end=nn_max, size=bin_width),
        xaxis='x5',
        hovertemplate=
            '<b>NN</b>: %{y:.3f}<br>'+
            '<b>Freq.</b>: %{x:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )

    
    
    #----------------
    # Set layout (and axes) for plot
    #–----------------
    
    layout = go.Layout(
            
        showlegend=False,
        autosize=True,
        # width=600,
        height=height_histo,
        margin={'l':0,'r':0,'t':40,'b':0},             
        hovermode='closest',
        bargap=0,
        title=title,
        titlefont={'family':'HelveticaNeue','size':18},        
        
        
        # NIB plot axes details
        xaxis1=dict(
            domain=domain_nib,
            showgrid=True,
            zeroline=False,
            range=[-0.5,nib_range[1]+0.5],
            tickvals = np.arange(nib_max),
            title='NIB'
        ),
                
        yaxis1=dict(
            domain=[0, 0.75],
            showgrid=True,
            zeroline=False,
            range=[nn_range[0], nn_range[1]],
            title='Average NN (s)'
        ),  
                
        yaxis2=dict(
            domain=[0.8,1],
            showgrid=True,
            # zeroline=False,
            title='Frequency'
        ),
        
        
    
        # VV plot axes details
        xaxis2=dict(
            domain=domain_vv,
            showgrid=True,
            zeroline=False,
            range=[vv_range[0], vv_range[1]],
            title='VV (s)'
        ),
                
        yaxis3=dict(
            domain=[0, 0.75],
            anchor='x2',
            showgrid=True,
            zeroline=False,
            range=[nn_range[0], nn_range[1]],
        ),
            
                
        yaxis4=dict(
            domain=[0.8,1],
            anchor='x2',
            showgrid=True,
            zeroline=True,
        ),
        
        
         # VN plot axes details
        xaxis3=dict(
            domain=domain_vn,
            showgrid=True,
            zeroline=False,
            range=[vn_range[0], vn_range[1]],
            title='VN (s)'
        ),
                
        yaxis5=dict(
            domain=[0, 0.75],
            anchor='x3',
            showgrid=True,
            zeroline=False,
            range=[nn_range[0], nn_range[1]],
        ),  
            
                
        yaxis6=dict(
            domain=[0.8,1],
            anchor='x3',
            showgrid=True,
            zeroline=False,
        ),
        
        
         # NV plot axes details
        xaxis4=dict(
            domain=domain_nv,
            showgrid=True,
            zeroline=False,
            range=[nv_range[0], nv_range[1]],
            title='NV (s)'
        ),
    
                
        yaxis7=dict(
            domain=[0, 0.75],
            anchor='x4',
            showgrid=True,
            zeroline=False,
            range=[nn_range[0], nn_range[1]],
        ),  
            
                
        yaxis8=dict(
            domain=[0.8,1],
            anchor='x4',
            showgrid=True,
            zeroline=False,
        ),
    
        xaxis5=dict(
            domain=domain_nn,
            showgrid=True,
            zeroline=False,
        ),
       
        
    )
    
    
    #------------------
    # Generate figure
    #--------------------
    
    data = [trace_hist2d_nib, trace_nib,
            trace_hist2d_vv, trace_vv,
            trace_hist2d_vn, trace_vn,
            trace_hist2d_nv, trace_nv,
            trace_nn]   
    fig = go.Figure(data=data, layout=layout)
    
    return fig




#------------------------
# Heartprint against NV
#–----------------------

def make_heartprint_nv(df_vv_stats,
					   nv_range = [0.3,0.7],
					   vv_range = [0,5],
					   nib_range = [0,10],
					   title = '',
                       bin_width = 0.008,
                       ):
    
    
    '''
    Function to make heartprint against NV
    
    Input:
        df_vv_stats: Dataframe containing VV interval statistics. 
            Has columns [Time (s), NIB, VV, VN , NV, NN avg]
            
		'var'_range: Plotting range for the statistic 'var'.
        		
		title: Title of the heartprint
    
        bin_width: Width of histogram bins (s)
            Take as a multiple of precision (0.004) to avoid aliasing
    
    Output:
        fig: Plotly figure of heartprint
    '''
    
    
    
   #-----------------
    # Fixed parameter values
    #–----------------
    
    height_histo = 400 # Height of histogram figures (fixed)
    bin_width = 0.01 # Width of each box of heatmap (s)

    
    # Relative widths
    width_nib = 1
    width_vv = 1
    width_nv = 0.3
    x_spacing = 0.2 # Horizontal distance between heat maps
    y_spacing = 0.05 # Distance between heat maps and histograms
    
    # x-domain sizes
    tot_width = width_nib + width_vv + width_nv + x_spacing + y_spacing
    domain_nib = np.array([0,width_nib/tot_width])
    domain_vv = domain_nib[1] + x_spacing/tot_width + np.array([0,width_vv])/tot_width
    domain_nv = domain_vv[1] + y_spacing/tot_width+ np.array([0,width_nv])/tot_width
    domain_nv[1]=1

    

        
    #–------------------
    # Traces for VV
    #–-------------------
    
    # Create normalised heatmap data
    vv_edges = np.arange(vv_range[0]-0.1, vv_range[1]+0.1,bin_width*4)
    nv_edges = np.arange(nv_range[0]-0.1, nv_range[1]+0.1,bin_width)  
    x=df_vv_stats['VV']
    y=df_vv_stats['NV']
    
    hist_data, vv_edges, nv_edges = np.histogram2d(x, y, bins=(vv_edges, nv_edges))
    hist_data = hist_data.T

    
    
    # Normalise each col (make sure that total occurence is non zero)
    max_occurences = hist_data.max(axis=0)
    hist_data_norm = np.array(
       [(hist_data[:,i]/max_occurences[i] if max_occurences[i] else hist_data[:,i]) for i in range(hist_data.shape[1])
        ]).T
    
    trace_hist2d_vv = go.Heatmap(
            x=vv_edges, 
            y=nv_edges, 
            z=hist_data_norm,
            name='density',
            xaxis='x2',
            yaxis='y3',
            colorscale='Blues',
            showscale=False,
            hovertemplate=
                '<b>VV</b>: %{x:.3f}<br>'+
                '<b>NV</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}),     
    )
      
    trace_vv = go.Histogram(
        x=df_vv_stats['VV'], 
        marker=dict(color='gray'),
        xbins=dict(start=vv_range[0], end=vv_range[1], size=bin_width*4),
        xaxis='x2',
        yaxis='y4',
        hovertemplate=
            '<b>VV</b>: %{x:.3f}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )
    
    
       
    #----------------
    # NIB traces
    #–---------------
    
    # Create normalised heatmap data
    nib_min,nib_max = [-0.5,20.5]
    nib_edges = np.arange(nib_min,nib_max,1)  
    x=df_vv_stats['NIB']
    y=df_vv_stats['NV']
    
    hist_data, nib_edges, nv_edges = np.histogram2d(x, y, bins=(nib_edges, nv_edges))
    hist_data = hist_data.T
    
    # Normalise each col (make sure that total occurence is non zero)
    max_occurences = hist_data.max(axis=0)
    hist_data_norm = np.array(
       [(hist_data[:,i]/max_occurences[i] if max_occurences[i] else hist_data[:,i]) for i in range(hist_data.shape[1])
        ]).T   
        
        
    # 2D histogram trace
    trace_hist2d_nib = go.Heatmap(
            x=nib_edges,
            y=nv_edges,
            z=hist_data_norm,
            xaxis='x1',
            yaxis='y1',
            colorscale='Blues',
            showscale=False,
            hovertemplate=
                '<b>NIB</b>: %{x:i}<br>'+
                '<b>NV</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}), 
    )
    

    # NIB histogram trace
    trace_nib = go.Histogram(
        x=df_vv_stats['NIB'], 
        marker=dict(color='gray'),
        xaxis='x1',
        yaxis='y2',
        xbins=dict(start=nib_min, end=nib_max, size=1),
        hovertemplate=
            '<b>NIB</b>: %{x:i}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),    
    )       
       
       
    # NV histogram trace 
    trace_nv = go.Histogram(
        y=df_vv_stats['NV'],
        marker=dict(color='gray'),
        ybins=dict(start=nv_range[0], end=nv_range[1], size=bin_width),
        xaxis='x3',
        hovertemplate=
            '<b>NV</b>: %{y:.3f}<br>'+
            '<b>Freq.</b>: %{x:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )

    
    
    
    #----------------
    # Set layout (and axes) for plot
    #–----------------
    
    layout = go.Layout(
            
        showlegend=False,
        autosize=True,
#         width=800,
        height=height_histo,
        margin={'l':0,'r':0,'t':40,'b':0},             
        hovermode='closest',
        bargap=0,
        title=title,
        titlefont={'family':'HelveticaNeue','size':18}, 
          
        # Plot axes details 
        
        #--------NIB axes----------
        xaxis1=dict(
            domain=domain_nib,
            showgrid=True,
            zeroline=False,
            range=[nib_range[0]-0.5,nib_range[1]],
            title='NIB'
        ),
                
        yaxis1=dict(
            domain=[0, 0.75],
            showgrid=True,
            zeroline=False,
            range=[nv_range[0], nv_range[1]],
            title='NV (s)'
        ),          
        
        yaxis2=dict(
        	anchor='x1',
        	domain=[0.8,1],
        	showgrid=True,
            title='Frequency',
        ),
        	
        	
        #--------VV axes----------
        xaxis2=dict(
            domain=domain_vv,
            showgrid=True,
            zeroline=False,
            range=[vv_range[0],vv_range[1]],
            title='VV (s)'
        ),
                
        yaxis3=dict(
            domain=[0, 0.75],
            anchor='x2',
            showgrid=True,
            zeroline=False,
            range=[nv_range[0], nv_range[1]],
        ),  
        
        yaxis4=dict(
            domain=[0.8, 1],
            anchor='x2',
            showgrid=True,
            # zeroline=False,
        ),         	
        	
        #-------NV axes----------
        xaxis3=dict(
            domain=domain_nv,
            showgrid=True,
        ),
                

        
    )
    
    
    #------------------
    # Generate figure
    #--------------------
    
    data = [trace_hist2d_nib,
            trace_nib,
			trace_hist2d_vv,
            trace_vv,
            trace_nv]   
    fig = go.Figure(data=data, layout=layout)
    
    return fig    
    
    
    
        

#------------------------
# Heartprint for VV vs NIB
#–----------------------

def make_heartprint_vv_nib(df_vv_stats,
					   vv_range = [0,5],
					   nib_range = [0,10],
					   title = '',
                       bin_width = 0.008,
                       ):
    
    
    '''
    Function to make heartprint for VV vs NIB
    
    Input:
        df_vv_stats: Dataframe containing VV interval statistics. 
            Has columns [Time (s), NIB, VV, VN , NV, NN avg]
            
		'var'_range: Plotting range for the statistic 'var'.
        		
		title: Title of the heartprint
        
        bin_width: Width of histogram bins (s)
            Take as a multiple of precision (0.004) to avoid aliasing            
    
    Output:
        fig: Plotly figure of heartprint
    '''
    
    
    
   #-----------------
    # Fixed parameter values
    #–----------------
    
    height_histo = 400 # Height of histogram figures (fixed)

    
    # Relative widths
    width_nib = 1
    width_vv = 0.3
    y_spacing = 0.05 # Distance between heat maps and histograms
    
    # x-domain sizes
    tot_width = width_nib + width_vv + y_spacing
    domain_nib = np.array([0,width_nib/tot_width])
    domain_vv = domain_nib[1] + y_spacing/tot_width+ np.array([0,width_vv])/tot_width
    domain_vv[1]=1
    
    
       
    #----------------
    # NIB traces
    #–---------------
    
    # Create normalised heatmap data
    nib_min,nib_max = [-0.5,20.5]
    nib_edges = np.arange(nib_min,nib_max,1)  
    vv_edges = np.arange(vv_range[0]-0.1, vv_range[1]+0.1,bin_width*4)

    x=df_vv_stats['NIB']
    y=df_vv_stats['VV']
    
    hist_data, nib_edges, vv_edges = np.histogram2d(x, y, bins=(nib_edges, vv_edges))
    hist_data = hist_data.T
    
    # Normalise each col (make sure that total occurence is non zero)
    max_occurences = hist_data.max(axis=0)
    hist_data_norm = np.array(
       [(hist_data[:,i]/max_occurences[i] if max_occurences[i] else hist_data[:,i]) for i in range(hist_data.shape[1])
        ]).T   
        
        
    # 2D histogram trace
    trace_hist2d_nib = go.Heatmap(
            x=nib_edges,
            y=vv_edges,
            z=hist_data_norm,
            xaxis='x1',
            yaxis='y1',
            colorscale='Blues',
            showscale=False,
            hovertemplate=
                '<b>NIB</b>: %{x:i}<br>'+
                '<b>VV</b>: %{y:.3f}<br>'+
                '<b>Freq.</b>: %{z:.3f}'+
                '<extra></extra>',
            hoverlabel=dict(
                bgcolor='gray',
                font={'color':'white'}), 
    )
    

    # NIB histogram trace
    trace_nib = go.Histogram(
        x=df_vv_stats['NIB'], 
        marker=dict(color='gray'),
        xaxis='x1',
        yaxis='y2',
        xbins=dict(start=nib_min, end=nib_max, size=1),
        hovertemplate=
            '<b>NIB</b>: %{x:i}<br>'+
            '<b>Freq.</b>: %{y:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),    
    )       
       
       
    # VV histogram trace 
    trace_vv = go.Histogram(
        y=df_vv_stats['VV'],
        marker=dict(color='gray'),
        ybins=dict(start=vv_range[0], end=vv_range[1], size=bin_width),
        xaxis='x2',
        hovertemplate=
            '<b>VV</b>: %{y:.3f}<br>'+
            '<b>Freq.</b>: %{x:i}'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='gray',
            font={'color':'white'}),     
    )

    
    
    
    #----------------
    # Set layout (and axes) for plot
    #–----------------
    
    layout = go.Layout(
            
        showlegend=False,
        autosize=True,
#         width=800,
        height=height_histo,
        margin={'l':0,'r':0,'t':40,'b':0},             
        hovermode='closest',
        bargap=0,
        title=title,
        titlefont={'family':'HelveticaNeue','size':18}, 
          
        # Plot axes details 
        
        #--------NIB axes----------
        xaxis1=dict(
            domain=domain_nib,
            showgrid=True,
            zeroline=False,
            range=[nib_range[0]-0.5,nib_range[1]],
            title='NIB'
        ),
                
        yaxis1=dict(
            domain=[0, 0.75],
            showgrid=True,
            zeroline=False,
            range=[vv_range[0], vv_range[1]],
            title='VV (s)'
        ),          
        
        yaxis2=dict(
        	anchor='x1',
        	domain=[0.8,1],
        	showgrid=True,
            title='Frequency',
        ),
        	
        	        	
        	
        #-------VV frequency axes----------
        xaxis2=dict(
            domain=domain_vv,
            showgrid=True,
        ),
                

        
    )
    
    
    #------------------
    # Generate figure
    #--------------------
    
    data = [trace_hist2d_nib,
            trace_nib,
            trace_vv]   
    fig = go.Figure(data=data, layout=layout)
    
    return fig    
    
    
        







#--------------------
# Scatter plots of VV, VN
#–----------------------
def make_scatter_vv_vn(df_vv_stats,
                  vv_range=False,
                  vn_range=[0.65,1.2],
                  title='Phase response curves'
                  ):
    
    
    '''
    Make scatter plots for NIB=1,2,3
    displaying VV vs VN
    with color denoting average NN
    
    Input:
        df_vv_stats: Dataframe of beat statistics with cols
            [Time (s), VV, NIB, NV, VN, NN_avg]
        regression: Include a linear regression on plots
        vv_range: Plot range for y axis. If false, determine automatically
        vn_range: Plot range for x axis
    
    Output:
        Plotly figure
    '''
    
    
    #--------------
    # Figure params
    #–--------------
   
    marker_size = 3
    
    fig_height = 350 # Height of histogram figures (fixed)
    
    # Relative widths
    width_nib1 = 1
    width_nib2 = 1
    width_nib3 = 1

    x_spacing = 0.2 # Horizontal distance between heat maps
    
    # x-domain sizes
    tot_width = width_nib1 + width_nib2 + width_nib3 + 2*x_spacing
    domain_nib1 = np.array([0,width_nib1/tot_width])
    domain_nib2 = domain_nib1[1] + x_spacing/tot_width + np.array([0,width_nib2])/tot_width
    domain_nib3 = domain_nib2[1] + x_spacing/tot_width + np.array([0,width_nib3])/tot_width
        
    # Add col for Time in (hr)
    df_vv_stats['Time (hr)'] = df_vv_stats['Time (s)']/3600
    
    #--------------
    # Traces for NIB = 1
    #–--------------
    
    # Select data with NIB=1
    df_nib1 = df_vv_stats[df_vv_stats['NIB']==1]
    
    # Define x and y variables
    x = df_nib1['VN'].values
    y = df_nib1['VV'].values
    
    # Determine VV range
    if len(y)==0:
        vv_range1=[0,10]
    else:
        vv_range1=vv_range if vv_range else [0,1.3*max(y)]
    
    # Compute regression line (as long as x,y nonempty)
    if len(x)>1:
        m,c,r,p,se1 = stats.linregress(x,y)
        # Regression text
        regression_text1='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          

            
        # Regression trace (still using scattergl so compatible with other traces)
        trace_regression1 = go.Scattergl(x=x, 
                                      y=m*x+c,
                                      mode='lines',
                                      marker=dict(color='black'),
                                      xaxis='x1',
                                      yaxis='y1',
                                      hoverinfo='skip',
                                      visible=False,
                                      )   
    else:
        regression_text1=''
        trace_regression1 = go.Scatter(xaxis='x1',
                                      yaxis='y1',
                                      hoverinfo='skip',
                                      visible=False,
                                      )
    
    
    
    # Make scatter trace
    trace_scatter1 = go.Scattergl(
        x=df_nib1['VN'],
        y=df_nib1['VV'],
        mode='markers',
        marker_color=df_nib1['NN avg'],                              
        marker=dict(
            size=marker_size,
            colorscale='viridis',
            showscale=True,
            coloraxis='coloraxis'),
        xaxis='x1',
        yaxis='y1',
        name='NIB=1',
        text=df_nib1['Time (hr)'],
        hovertemplate =
            '<b>VV</b>: %{y:.3f}<br>'+
            '<b>VN</b>: %{x:.3f}<br>'+
            '<b>NN</b>: %{marker.color:.3f}<br>'+
            '<b>Time</b>: %{text:.4f}hr'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='purple',
            font={'color':'white'}),
        )


    
    #--------------
    # Traces for NIB = 2
    #–--------------
    
    # Select data with NIB=1
    df_nib2 = df_vv_stats[df_vv_stats['NIB']==2]
    
    # Define x and y variables
    x = df_nib2['VN'].values
    y = df_nib2['VV'].values
    
    # Determine VV range
    if len(y)==0:
        vv_range2=[0,10]
    else:
        vv_range2=vv_range if vv_range else [0,1.3*max(y)]
    
    # Compute regression line (as long as x,y nonempty)
    if len(x)>1:
        m,c,r,p,se1 = stats.linregress(x,y)
        # Regression text
        regression_text2='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          

            
        # Regression trace
        trace_regression2 = go.Scattergl(x=x, 
                                      y=m*x+c,
                                      mode='lines',
                                      marker=dict(color='black'),
                                      xaxis='x2',
                                      yaxis='y2',
                                      hoverinfo='skip',
                                      visible=False,
                                      )   
    else:
        regression_text2=''
        trace_regression2 = go.Scatter(xaxis='x2',
                                      yaxis='y2',
                                      hoverinfo='skip',
                                      visible=False,
                                      )
    
    
    # Make scatter trace
    trace_scatter2 = go.Scattergl(
        x=df_nib2['VN'],
        y=df_nib2['VV'],
        mode='markers',
        marker_color=df_nib2['NN avg'],                              
        marker=dict(
            size=marker_size,
            colorscale='viridis',
            showscale=True,
            coloraxis='coloraxis'),
        xaxis='x2',
        yaxis='y2',
        name='NIB=2',
        text=df_nib2['Time (hr)'],
        hovertemplate =
            '<b>VV</b>: %{y:.3f}<br>'+
            '<b>VN</b>: %{x:.3f}<br>'+
            '<b>NN</b>: %{marker.color:.3f}<br>'+
            '<b>Time</b>: %{text:.4f}hr'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='purple',
            font={'color':'white'}),
        )
    
    
    
        
    #--------------
    # Traces for NIB = 3
    #–--------------
    
    # Select data with NIB=1
    df_nib3 = df_vv_stats[df_vv_stats['NIB']==3]
    
    # Define x and y variables
    x = df_nib3['VN'].values
    y = df_nib3['VV'].values
    
    # Determine VV range
    if len(y)==0:
        vv_range3=[0,10]
    else:
        vv_range3=vv_range if vv_range else [0,1.3*max(y)]

    # Compute regression line (as long as x,y nonempty)
    if len(x)>1:
        m,c,r,p,se1 = stats.linregress(x,y)
        # Regression text
        regression_text3='y={}x+{},  R^2={}'.format(round(m,2),round(c,2),round(r**2,2))          
            
        # Regression trace
        trace_regression3 = go.Scattergl(x=x, 
                                      y=m*x+c,
                                      mode='lines',
                                      marker=dict(color='black'),
                                      xaxis='x3',
                                      yaxis='y3',
                                      hoverinfo='skip',
                                      visible=False,
                                      )   
    else:
        regression_text3=''
        trace_regression3 = go.Scatter(xaxis='x3',
                                      yaxis='y3',
                                      hoverinfo='skip',
                                      visible=False,
                                      )
    
    # Make scatter trace
    trace_scatter3 = go.Scattergl(
        x=df_nib3['VN'],
        y=df_nib3['VV'],
        mode='markers',
        marker_color=df_nib3['NN avg'],                              
        marker=dict(
            size=marker_size,
            colorscale='viridis',
            showscale=True,
            coloraxis='coloraxis'),
        xaxis='x3',
        yaxis='y3',
        name='NIB=3',
        text=df_nib3['Time (hr)'],
        hovertemplate =
            '<b>VV</b>: %{y:.3f}<br>'+
            '<b>VN</b>: %{x:.3f}<br>'+
            '<b>NN</b>: %{marker.color:.3f}<br>'+
            '<b>Time</b>: %{text:.4f}hr'+
            '<extra></extra>',
        hoverlabel=dict(
            bgcolor='purple',
            font={'color':'white'}),
        )
    
    
    #--------------
    # Figure layout and axes
    #---------------
    
    # Annotations for linear regression equation
    regression_annotations = [    
        # Regression text for NIB=1
        dict(
            x=sum(vn_range)/2,
            y=0.2,
            text=regression_text1,
            xref='x1',
            yref='y1',
            showarrow=False,
            font = dict(
                    color = "black",
                    size = 14)
            ),
        # Regression text for NIB=2
        dict(
            x=sum(vn_range)/2,
            y=0.2,
            text=regression_text2,
            xref='x2',
            yref='y2',
            showarrow=False,
            font = dict(
                    color = "black",
                    size = 14)
            ),
        # Regression text for NIB=3
        dict(
            x=sum(vn_range)/2,
            y=0.2,
            text=regression_text3,
            xref='x3',
            yref='y3',
            showarrow=False,
            font = dict(
                    color = "black",
                    size = 14)
            ),    
    ]
    
    nib_text_annotations = [
            # NIB =1 text
            dict(
                x=vn_range[0]+0.25,
                y=vv_range1[1]*0.95,
                text='NIB=1',
                xref='x1',
                yref='y1',
                showarrow=False,
                font = dict(
                        color = "black",
                        size = 14)
                ),

            # NIB =2 text
            dict(
                x=vn_range[0]+0.25,
                y=vv_range2[1]*0.95,
                text='NIB=2',
                xref='x2',
                yref='y2',
                showarrow=False,
                font = dict(
                        color = "black",
                        size = 14)
                ),
            
            # NIB =3 text
            dict(
                x=vn_range[0]+0.25,
                y=vv_range3[1]*0.95,
                text='NIB=3',
                xref='x3',
                yref='y3',
                showarrow=False,
                font = dict(
                        color = "black",
                        size = 14)
                ),            
  
            ]       
    
    
    
    layout = go.Layout(
            
        showlegend=False,
        autosize=True,
        # width=600,
        height=fig_height,
        margin={'l':0,'r':0,'t':40,'b':0},             
        hovermode='closest',
        bargap=0,
        title=title,
        annotations=nib_text_annotations,
        coloraxis_colorbar=dict(
                              title='NN (s)'),
   
        # NIB1 plot axes details
        xaxis1=dict(
            domain=domain_nib1,
            showgrid=True,
            zeroline=False,
            range=vn_range,
            # tickvals = np.arange(nib_range[1]+0.5),
            title='VN (s)'
        ),
                
        yaxis1=dict(
            domain=[0, 1],
            showgrid=True,
            zeroline=False,
            range=vv_range1,
            title='VV (s)'
        ),  
        
        
    
        # NIB2 plot axes details
        xaxis2=dict(
            domain=domain_nib2,
            showgrid=True,
            zeroline=False,
            range=vn_range,
            title='VN (s)'
        ),
                
        yaxis2=dict(
            domain=[0, 1],
            anchor='x2',
            showgrid=True,
            zeroline=False,
            range=vv_range2,
        ),
            
        
        
         # NIB3 plot axes details
        xaxis3=dict(
            domain=domain_nib3,
            showgrid=True,
            zeroline=False,
            range=vn_range,
            title='VN (s)'
        ),
                
        yaxis3=dict(
            domain=[0, 1],
            anchor='x3',
            showgrid=True,
            zeroline=False,
            range=vv_range3,
        ),  
            

        
    )    
    
    #------------------
    # Generate figure
    #--------------------
    
    data = [trace_scatter1, trace_scatter2, trace_scatter3,
            trace_regression1, trace_regression2, trace_regression3]

    fig = go.Figure(data=data, layout=layout)
    
    
    #---------------
    # Add a button to turn on and off scatter plot
    #---------------
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.65,
                y=1.2,
                buttons=list([
                    dict(label="Regression Off",
                          method="update",
                          args=[{"visible": [True, True, True, False,False,False]},
                                {"annotations": nib_text_annotations}]
                          ),
                    dict(label="Regression On",
                          method="update",
                          args=[{"visible": [True, True, True, True, True, True]},
                                {"annotations": regression_annotations+nib_text_annotations}]
                          ),                
                    ]),
            )
        ])    
    
    
    
    
    
    

    return fig







#-------------------
# Proportion of PVCs as a function of NN avg - histogram
#–-----------------
def make_prop_pvc_hist(df_vv_stats,
					   nn_range=[0.4,1.4],
					   nn_interval_width=0.05,
                       title=''):
	'''
	Plot a histogram of the proportion of PVCs as a function of NN avg
	
	Input:
		df_vv_stats: DataFrame with cols
			[Time (hr), NIB, NV, VN, VV, NN avg]
	
	Output:
		fig_prop_pvc: plotly figure
	
	'''
	


	nn_avg_edges = np.arange(nn_range[0],nn_range[1]+0.01,nn_interval_width)

	
	# Initalise lists
	list_prop_pvc = []

	for i in range(len(nn_avg_edges)-1):
		nn_lower = nn_avg_edges[i]
		nn_upper = nn_avg_edges[i+1]
	
		# Select portion of df that contains nn avg within range
		df_select = df_vv_stats[(df_vv_stats['NN avg']>=nn_lower)&(df_vv_stats['NN avg']<nn_upper)]
	
		# Compute total number of normal beats
		total_sinus = df_select['NIB'].sum()
		# Compute total number of ectopic beats
		total_ectopic = len(df_select)
		# Compute proportion ectopic (if beats exist)
		if len(df_select)>0:
			prop_ectopic = (total_ectopic)/(total_sinus+total_ectopic)
		else:
			prop_ectopic = np.nan
		# Add prop_ectopic to list
		list_prop_pvc.append(prop_ectopic)

	
	# Put into datafrmae
	df_prop_pvc = pd.DataFrame(
		{'NN lower': nn_avg_edges[:-1],
		 'NN upper': nn_avg_edges[1:],
		 '%PVC': list_prop_pvc})

	# Create NN avg column
	df_prop_pvc['NN avg'] = (df_prop_pvc['NN lower']+df_prop_pvc['NN upper'])/2



    # Create plot
	trace_bar = go.Bar(x=df_prop_pvc['NN avg'], 
					   y=df_prop_pvc['%PVC'],
# 					   marker=dict(color='gray'),
					   width=nn_interval_width,
					   customdata=df_prop_pvc[['NN lower','NN upper']].values,
					   hovertemplate=
						   '<b>NN interval</b>: [%{customdata[0]:.2f},%{customdata[1]:.2f}]<br>'+
							'<b>%PVC.</b>: %{y:.3f}'+
							'<extra></extra>',
						hoverlabel=dict(
							bgcolor='gray',
							font={'color':'white'}),                       )

	layout = go.Layout(width=300,
					   height=300,
					   margin={'l':0,'r':0,'t':40,'b':0},
                       title=title,
				   
					   xaxis=dict(
						   title='NN (s)'
					   ),
					   yaxis=dict(
						   title='%PVC'
					   ),
	)

	fig_prop_pvc = go.Figure(trace_bar,
							 layout,
	)


	return fig_prop_pvc




def make_pvc_vs_nn_curated(df_pvc_vs_nn,
                           title='',
                           beat_threshold = 100
                           ):
    
    '''
	Plot a histogram of the proportion of PVCs as a function of NN avg
    Here we use data that has already been curated for the plot
    This uses a much smaller file size and plot generates faster.
	
	Input:
		df_pvc_vs_nn: DataFrame with cols
			[NN avg, %PVC, Total beats]
        beat_threshold: number of beats required in bin in order to
            be displayed in histogram.
	
	Output:
		fig_prop_pvc: plotly figure
	
	'''
    
    # Only plot data that exceeds beat threshold within bins
    df_pvc_vs_nn = df_pvc_vs_nn[df_pvc_vs_nn['Total beats']>beat_threshold]

    # Create plot
    trace_bar = go.Bar(x=df_pvc_vs_nn['NN avg'], 
                       y=df_pvc_vs_nn['%PVC'],
					   customdata=df_pvc_vs_nn['Total beats'].values,            
                       hovertemplate=
                           '<b>NN avg</b>: %{x:.3f}<br>'+
                           '<b>%PVC</b>: %{y:.3f}<br>'+
                           '<b>Total beats</b>: %{customdata:.i}<br>'+
                           '<extra></extra>',
                       hoverlabel=dict(bgcolor='gray',
                                       font={'color':'white'}
                                       ),
    )
    
    
    
    layout = go.Layout(height=300,
                       autosize=True,
                       bargap=0,
                       margin = {'l':0,'r':0,'t':40,'b':0},
                       title=title,
                       titlefont={'family':'HelveticaNeue','size':18},   
    				   
    					   xaxis=dict(
    						   title='NN (s)',
                               range=[0.3,1.4],
    					   ),
    					   yaxis=dict(
    						   title='%PVC'
    					   ),
    	)
    
    fig_pvc_vs_nn = go.Figure(trace_bar,
       							  layout,
    )
    
    
    return fig_pvc_vs_nn






#–-----------------------
# Histogram of stats as a function of hour in the day 
#–----------------------

def make_hist_24hour(df_vv_stats, 
                     hookup_time=0,
                     pvc_range=[0,0.6],
                     m_range=[0.4,1.4],
                     nn_range=[0.4,1.4],
                     nv_range=[0,1.5],
                     title='',
                     ):
    '''
    Make row of plots containing info on NN, %PVCs VV, and m over a 24 hour period
    Input:
        df_vv_stats: DataFrame with cols
            [Time (hr), NIV, VV, NV, VN, NN avg]
        hookup_time: Hour at which Holter recording begins
    '''
    
    # If hookuptime is nan, set to 0
    if np.isnan(hookup_time):
        hookup_time=0
    
	# Function to get proportion of PVCs from (subset of) df_vv_stats
    def get_prop_pvc(df_vv_stats):
        # Number of V beats is number of entries in df
        num_v = len(df_vv_stats)
		# Number of N beats is total of NIB column
        num_n = df_vv_stats['NIB'].sum()
	    # Proportion of V beats (if beats exist)
        if len(df_vv_stats)>0:
            prop_v = num_v/(num_n + num_v)
        else: prop_v = np.nan
	
        return prop_v
		
    # Function to get gradient of VV-VN scatter plot
    def get_grad_vv_vn(df_vv_stats):

        # Get cases of NIB=1
        df_nib1 = df_vv_stats[df_vv_stats['NIB']==1]
    
        # Define x and y variables of VV-VN plot
        x = df_nib1['VN'].values
        y = df_nib1['VV'].values

        # Compute linear regression line, as long as more than nmin points
        nmin = 20
        if len(x)>nmin:
            m,c,r,p,se1 = stats.linregress(x,y)
        else:
            m,c,r,p,se1 = [np.nan]*5
        
        # Return gradient of regression
        return m 

    

    # Make time columns specific to Holter start time
    df_vv_stats['Time shifted (hr)'] = df_vv_stats['Time (hr)']+hookup_time
    df_vv_stats['Time of day (hr)'] = df_vv_stats['Time shifted (hr)']%24
    df_vv_stats['Hour of day'] = df_vv_stats['Time of day (hr)'].apply(np.floor)
    df_vv_stats['Day number'] = (df_vv_stats['Time (hr)']+hookup_time)//24
    

    # Initialise lists to compute hourly stats
    list_prop_v = []
    list_grad_vv_vn = []
    list_hour_of_day = []
    
       
    # Loop through each hour of Holter recording
    for t_start in np.arange(int(df_vv_stats['Time shifted (hr)'].iloc[1]),
                             df_vv_stats['Time shifted (hr)'].iloc[-1]):
        # End of hour interval
        t_end = t_start + 1
        # Get stats within the hour
        df_vv_stats_hour = \
            df_vv_stats[(df_vv_stats['Time shifted (hr)']>=t_start)&\
                        (df_vv_stats['Time shifted (hr)']<t_end)]
        # Compute proportion of PVCs within hour interval
        prop_v = get_prop_pvc(df_vv_stats_hour)
        list_prop_v.append(prop_v)
        
        # Compute gradient of VV-VN
        grad_vv_vn = get_grad_vv_vn(df_vv_stats_hour)
        list_grad_vv_vn.append(grad_vv_vn) 
        
        # Record hour of day
        if not df_vv_stats_hour.empty:
            hour_of_day = df_vv_stats_hour['Hour of day'].iloc[0]
        else:
            hour_of_day = np.nan
        list_hour_of_day.append(hour_of_day)

        
    # Build dataframe
    df_hour_stats = pd.DataFrame({'Hour of day':list_hour_of_day,
                       '%PVC': list_prop_v,
                       'm':list_grad_vv_vn})
    
    
    #------------Build figure-------------#

    # Include tooltips or not
    hoverinfo = 'skip'

    # Relative widths
    width_hist = 1
    x_spacing = 0.3 # Horizontal distance between heat maps
    
    # x-domain sizes
    tot_width = 4*width_hist+ 4*x_spacing 
    domain_nn = np.array([0,width_hist/tot_width])
    domain_pvc = domain_nn[1] + x_spacing/tot_width + np.array([0,width_hist])/tot_width
    domain_nv = domain_pvc[1] + x_spacing/tot_width+ np.array([0,width_hist])/tot_width
    domain_m = domain_nv[1] + x_spacing/tot_width+ np.array([0,width_hist])/tot_width
 
    # X-axes label
    xlabel = 'Hour of day' if hookup_time!=0 else 'Hour (start time unknown)'
    
    trace_nn = go.Box(
        x=df_vv_stats['Hour of day'],
        y=df_vv_stats['NN avg'],
        xaxis='x1',
        yaxis='y1',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )    
    

    trace_pvc = go.Box(
        x=df_hour_stats['Hour of day'],
        y=df_hour_stats['%PVC'],
        xaxis='x2',
        yaxis='y2',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )
    
    trace_vv = go.Box(
        x=df_vv_stats['Hour of day'],
        y=df_vv_stats['NV'],
        xaxis='x3',
        yaxis='y3',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )    
    
    trace_m = go.Box(
        x=df_hour_stats['Hour of day'],
        y=df_hour_stats['m'],
        xaxis='x4',
        yaxis='y4',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )
    
    
 
    layout = go.Layout(
        height=250,
        showlegend=False,
        margin={'l':10,'r':10,'t':60,'b':40},
        title=title,
        
        xaxis1=dict(range=[-1,23.9],
                   domain=domain_nn,
                   tick0=0,
                   dtick=6,
                   title=xlabel,
        ),
        
        yaxis1=dict(title='NN',
                    anchor='x1',
                    range=nn_range,
                    title_standoff = 5,
        ),
        
        
        xaxis2=dict(range=[-1,23.9],
                    domain=domain_pvc,
                    tick0=0,
                    dtick=6,
                    title=xlabel,
        ),
        
        
        yaxis2=dict(anchor='x2',
                    title='%PVC',
                    range=pvc_range, 
                    title_standoff = 0,
        ),
        
        
        xaxis3=dict(range=[-1,23.9],
                    domain=domain_nv,
                    tick0=0,
                    dtick=6,
                    title=xlabel,
        ),
        
        
        yaxis3=dict(anchor='x3',
                    title='NV',
                    range=nv_range, 
                    title_standoff = 0,
        ),
        
        xaxis4=dict(range=[-1,23.9],
                    domain=domain_m,
                    tick0=0,
                    dtick=6,
                    title=xlabel,
        ),   
        
        yaxis4=dict(anchor='x4',
                    title='m',
                    range=m_range,  
                    title_standoff = 0,
        ),        
    )
    
    
    fig = go.Figure(
        data=[trace_nn, trace_pvc, trace_vv, trace_m],
        layout=layout,
    )
    
    return fig    




def make_hist_24hour_curated(df_24hour_stats, 
                             hookup_time,
                             pvc_range=[-0.05,0.7],
                             m_range=[0.4,1.4],
                             nn_range=[0.4,1.4],
                             nv_range=[0,1.5],
                             title='Statistics as a function of hour in the day',
                             ):
    '''
    Make row of plots containing info on NN, %PVCs VV, and m over a 24 hour period
    Uses pre-curated stats for faster run time.
    Input:
        df_24hour_stats: DataFrame with cols
            [Hour of day, Stat type, %PVC, m, NN avg, NV]
        hookup_time: Hour at which Holter recording begins
    '''


    
    #------------Build figure-------------#

    # Include tooltips or not
    hoverinfo = 'skip'

    # Relative widths
    width_hist = 1
    x_spacing = 0.3 # Horizontal distance between heat maps
    
    # x-domain sizes
    tot_width = 4*width_hist+ 4*x_spacing 
    domain_nn = np.array([0,width_hist/tot_width])
    domain_pvc = domain_nn[1] + x_spacing/tot_width + np.array([0,width_hist])/tot_width
    domain_nv = domain_pvc[1] + x_spacing/tot_width+ np.array([0,width_hist])/tot_width
    domain_m = domain_nv[1] + x_spacing/tot_width+ np.array([0,width_hist])/tot_width
 
    # X-axes label
    xlabel = 'Hour (start time unknown)' if np.isnan(hookup_time) else 'Hour of day'
    
    trace_nn = go.Box(
        x=df_24hour_stats['Hour of day'],
        y=df_24hour_stats['NN avg'],
        xaxis='x1',
        yaxis='y1',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )    
    

    trace_pvc = go.Box(
        x=df_24hour_stats['Hour of day'],
        y=df_24hour_stats['%PVC'],
        xaxis='x2',
        yaxis='y2',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )
    
    trace_vv = go.Box(
        x=df_24hour_stats['Hour of day'],
        y=df_24hour_stats['NV'],
        xaxis='x3',
        yaxis='y3',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )    
    
    trace_m = go.Box(
        x=df_24hour_stats['Hour of day'],
        y=df_24hour_stats['m'],
        xaxis='x4',
        yaxis='y4',
        boxpoints=False,
        hoverinfo=hoverinfo,
    )
    
    
 
    layout = go.Layout(
        height=250,
        showlegend=False,
        margin={'l':10,'r':10,'t':60,'b':40},
        title=title,
        titlefont={'family':'HelveticaNeue','size':18},  
        
        xaxis1=dict(range=[-1,23.9],
                   domain=domain_nn,
                   tick0=0,
                   dtick=6,
                   title=xlabel,
        ),
        
        yaxis1=dict(title='NN',
                    anchor='x1',
                    range=nn_range,
                    title_standoff = 5,
        ),
        
        
        xaxis2=dict(range=[-1,23.9],
                    domain=domain_pvc,
                    tick0=0,
                    dtick=6,
                    title=xlabel,
        ),
        
        
        yaxis2=dict(anchor='x2',
                    title='%PVC',
                    range=pvc_range, 
                    title_standoff = 0,
        ),
        
        
        xaxis3=dict(range=[-1,23.9],
                    domain=domain_nv,
                    tick0=0,
                    dtick=6,
                    title=xlabel,
        ),
        
        
        yaxis3=dict(anchor='x3',
                    title='NV',
                    range=nv_range, 
                    title_standoff = 0,
        ),
        
        xaxis4=dict(range=[-1,23.9],
                    domain=domain_m,
                    tick0=0,
                    dtick=6,
                    title=xlabel,
        ),   
        
        yaxis4=dict(anchor='x4',
                    title='m',
                    range=m_range,  
                    title_standoff = 0,
        ),        
    )
    
    
    fig = go.Figure(
        data=[trace_nn, trace_pvc, trace_vv, trace_m],
        layout=layout,
    )
    
    return fig    


  




#-----------------
# Regression gradient over time
#-------------------

def make_m_vs_time(df_regression,
                   title='',
                   yrange=[-0.05,1.6],
                   xrange=[-6,174]):
    '''
    Plot the VV-VN regression gradient over time
    
    Input:
        df_regression: DataFrame with cols
            [m, c, r2, NN avg, VV avg, Time (s)]
        title: plot title
    Output:
        Plotly fig
            
    '''
    
    # Make column for time in hours
    df_regression['Time (hr)'] = df_regression['Time (s)']/3600
    
    fig = px.scatter(df_regression, 
                      x="Time (hr)",
                      y="m", 
                      color='r2',
                      range_color=[0.5,1],
                      color_continuous_scale=px.colors.diverging.Portland,
                      title=title,
    )
    
    fig.update_layout(
        xaxis = dict(
            range=xrange,
            tick0 = 0,
            dtick = 24,
        ),
        yaxis = dict(
            range=yrange,
        ),        
        font = dict(size=12),
        margin=dict(l=10,r=10,b=10,t=40)
    )
    
    fig.update_traces(marker=dict(size=3))

    
    return fig












