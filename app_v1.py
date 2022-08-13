#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 27, 2022

App v1

@author: Thomas M. Bury

"""



from dash import Dash, html, dcc
from dash.dependencies import Input, Output

css = 'https://codepen.io/chriddyp/pen/bWLwgP.css'
app = Dash(__name__, external_stylesheets=[css])
server = app.server

#################################
# App details go in here

app.layout = html.Div([
    # List all components of app here
    html.H1('ECG Dashboard'),
])

#################################

if __name__ == '__main__':
    app.run_server(debug=True)