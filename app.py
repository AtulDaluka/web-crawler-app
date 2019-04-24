import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
"""
Created on Fri Feb 8 11:50:08 2019

@author: atul.daluka
"""
import os
import numpy as np
import io
import pandas as pd
import plotly.plotly as py
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
# import dash_table_experiments as dt
import dash_table as dt
import base64
import requests
import time
from bs4 import BeautifulSoup
import urllib.request as ur
import urllib.parse
import urllib
import bs4
import csv
from datetime import datetime
import re
import regex
import logging
from multiprocessing import Pool
from pandas.compat import StringIO
import json
import ast
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
import sys
from langdetect import detect
from money_parser import price_str
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import argparse
import numpy
import six
#Topic Modeling on 120K URLs Using LDA(Mallet)-Based Machine Learning Algorithm
#Step 1. Import all the necessary packages. 
import sklearn
import string
# import nltk
from pprint import pprint
# import matplotlib.pyplot as plt

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import warnings
import plotly
import datetime
from random import random
import copy

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\a.daluka\\Documents\\Google_API\\Category_Classification-b7ddf209f440.json"

beers=['Chesapeake Stout', 'Snake Dog IPA', 'Imperial Porter', 'Double Dog IPA']

bitterness = go.Bar(
    x=beers,
    y=[35, 60, 85, 75],
    name='IBU',
    marker={'color':'red'}
)
alcohol = go.Bar(
    x=beers,
    y=[5.4, 7.1, 9.2, 4.3],
    name='ABV',
    marker={'color':'blue'}
)

beer_data = [bitterness, alcohol]
beer_layout = go.Layout(
    barmode='group',
    title = 'Beer Comparison'
)

beer_fig = go.Figure(data=beer_data, layout=beer_layout)

########### Display the chart

app = dash.Dash()
server = app.server

app.layout = html.Div(children=[
    html.H1('Flying Dog Beers'),
    dcc.Graph(
        id='flyingdog',
        figure=beer_fig
    )]
)

if __name__ == '__main__':
    app.run_server()
