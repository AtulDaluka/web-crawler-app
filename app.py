"""
Created on Fri Feb 8 11:50:08 2019

@author: atul.daluka
"""
import flask
import os
import numpy as np
import io
import pandas as pd
import plotly.plotly as py
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
# import dash_table_experiments as dt
import dash_table as dt
import base64
import requests
import time
import flask
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
from selenium import webdriver
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
import nltk
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
import numpy as np
import argparse
import cv2
# Enable logging for gensim - optional
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from gensim import corpora
# import en_core_web_lg as en 
from nltk.corpus import stopwords
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\a.daluka\\Documents\\Google_API\\Category_Classification-b7ddf209f440.json"
import dash_auth
import plotly
from matplotlib.pyplot import specgram
# import keras
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.layers import Flatten, Dropout, Activation
# from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
# from sklearn.metrics import confusion_matrix
# from keras import regularizers
# from keras.utils import np_utils
# from sklearn.preprocessing import LabelEncoder
# import scipy.io.wavfile
# from keras.models import model_from_json
import datetime
from random import random
import copy
# from AudA import page_1
import plotly.graph_objs as go

login_success = 0

# Setup app
app = dash.Dash(__name__)
server = app.server
server.secret_key = os.environ.get('secret_key', 'secret')

external_css = ["https://fonts.googleapis.com/css?family=Overpass:300,300i",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/dab6f937fd5548cebf4c6dc7e93a10ac438f5efb/dash-technical-charting.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })

app.config.suppress_callback_exceptions = True

app.scripts.config.serve_locally = True

colors = {
    'background': '#C3F1F7',
    'text': '#09161B',
    'graphground': '#000000',
    'graphtext': '#FFFFFF'
}

mark1='''
Identifies presence of advertising on digital ad platforms
with behavior and conversion tracking pixels / tag managers
'''
mark2='''
Harvests common business attributes by sourcing
firmographics data from listing sites with structured data, e.g. Wiki, LinkedIn, Yelp, etc.
'''
mark3='''
Identifies customers with eComm objectives like
eComm platforms, cart/checkout options, payment channels
'''
mark4='''
Identifies customers with app links on websites and
scraps attributes from app store pages Detects app links on websites
'''
mark5='''
Determines brand positioning in category,
the currency and average price points of items sold on the SMB websites
'''
mark6='''
Identifies businesses with content publisher objectives, their inbound 
and outbound digital advertisement tech like Google Ad Sense.
'''
mark7='''
Identifies customers with lead gen objectives,
their presence of sign-up forms, including email subscription fields, “nearby store” maps, etc…
'''
# mark8='''
# Identifies customers with content publisher objectives like
# Count the number of ads on a page, such as Google AdSense, MediaNet, etc.
# '''
mark8='''
AI driven image detection and object recognition functionality to
capture more deeper insights about businesses, their customer segements and products.
'''
mark9='''
Identifying presence of social media websites on business web pages and
scrape Attributes from Instagram, twitter profile of businesses.
'''
mark10='''
Using NLP and AI based techniques to archetype leads into categories
by scraping texts from web pages, translating to English if needed and
building unsupervised topic modelling to automatically categorize the businesses.
'''
mark11='''
Leveraging Google Cloud Services to translate text and sub-categorize
leads as Ecommerce/Business Services/App/Food Restaurants based on the information on their web pages
'''
mark12='''
Artificially Intelligent automated archetyping
based on Google API and Topic Modelling results and creating
intelligent business rules to handle conflict between two of the mechanisms
'''
tab3_layout = html.Div([
    html.Div([
            html.Span("Web Crawler Playbook", className='app-title'),
            html.Div(
                html.Img(src=app.get_asset_url('download.jfif'),height="100%"),style={"float":"right","height":"100%"}),

            
                     
            html.Div(dcc.Link(
                html.Button(id='logout_button', type='logout', children='logout',style={'color':'white','fontWeight':'bold','backgroundColor':'darkblue'})
                ,href='http://127.0.0.1:5000/'),style={"float":"right",'marginTop': 15,'marginRight':10}),

##            html.Div([html.Div(
##                html.Button(id='reset_button', type='reset', children='reset',style={'color':'black','fontWeight':'bold','backgroundColor':'lightblue'})
##                     ,style={"float":"right",'marginTop': 15,'marginRight':10}),html.Div(id='page_content2')])

            html.Div(dcc.Link(
                html.Button(id='reset_button', type='reset', title='To have a fresh start of application, press the button! Remember, all the data on app will be lost!',children='reset',style={'color':'white','fontWeight':'bold','backgroundColor':'lightblue'})
                ,href='/web-crawler/app/logged-in'),style={"float":"right",'marginTop': 15,'marginRight':10}),
            ],
            className="row header"
            ),
    dcc.Tabs(id="tabs", children=[
    dcc.Tab(label='Home Page',children=[
        html.Div([ html.Div([
            html.Div(
                        [
                            html.P("PIXEL LOCATOR", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture1.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark1)],style={'fontSize':11,}),
                                                
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':50, 'marginBottom':10}
                    ),
            html.Div(
                        [
                            html.P("FIRMOGRAPHIC MINER", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture2.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark2)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':50, 'marginBottom':10}
                        ),
            html.Div(
                        [
                            html.P("ECOMM DETECTOR", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture3.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark3)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),
            html.Div(
                        [
                            html.P("APPS STORE MINER", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture4.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark4)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),

               
            ],
            className="rows",
            style={'columnCount':4, 'marginBottom':20, 'marginTop':20,'marginLeft':40}
        ),
                                html.Div([
            html.Div(
                        [
                            html.P("PRICE POINT CALCULATOR", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture5.jpg'),height="100%"),
                            html.Div([dcc.Markdown(children=mark5)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':20}
                        ),
            html.Div(
                        [
                            html.P("AD TECH & AD COUNT", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture6.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark6)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),
            
            html.Div(
                        [
                            html.P("SIGN-UP FORM DETECTOR", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture7.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark7)],style={'fontSize':11,'float':'center'}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),
            html.Div(
                        [
                            html.P("IMAGE DETECTOR", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture8.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark8)],style={'fontSize':11, 'float':'center'}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),
                
            
            ],
            className="rows",
            style={'columnCount':4, 'marginBottom':20, 'marginTop':10,'marginLeft':40}
        ),
             html.Div([
            html.Div(
                        [
                            html.P("SOCIAL PRESENCE", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture1.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark9)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':20}
                        ),
            html.Div(
                        [
                            html.P("TOPIC MODELING", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture2.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark10)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),
            
            html.Div(
                        [
                            html.P("GOOGLE API", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture3.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark11)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':20}
                        ),
            html.Div(
                        [
                            html.P("AI AUTOMATED ARCHETYPE", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'}),
                            html.Img(src=app.get_asset_url('Picture6.png'),height="100%"),
                            html.Div([dcc.Markdown(children=mark12)],style={'fontSize':11}),
                            
                        ],
                        className="ten columns",
                        style={'columnCount':1, 'rowCount':30,'marginBottom':10}
                        ),
                
            
            ],
            className="rows",
            style={'columnCount':4, 'marginBottom':20, 'marginTop':10,'marginLeft':40}
        )
        ])]),

    dcc.Tab(label='Application View Page', children=[
        html.Div([html.P('Upload List of Websites ("HTTP://WWW.XYZ.COM"):'),
                dcc.Upload(
                id='upload-data-file',
                children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File')]),
                style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
                },
                    # Do not Allow multiple files to be uploaded
                    multiple=False
                ),
                # html.Div(id='output_div_1'),
                html.Div(id='datatable-1'),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),
                html.P('or',style={'textAlign': 'center','fontWeight':'bold','fontSize':30}),
                html.Div([
                    html.Div(
                        dcc.Input(
                id = 'input_button_1',
                placeholder='Enter a value...',
                type='text',
                value='',
                style={'width': '100%'}
                ),className="eleven columns"),
                html.Div(html.Button(id='submit-button-1', type='submit', children='Submit',style={'color':'white','backgroundColor':'#506784'}),
                         style={'float':'right'})],className='row'),
                html.Div(id='output_div_2'),
                ],style={'margin-top': '20'}),
        html.Br(),

        html.Div([html.P('Filter by solution type:'),
                  html.Div([html.Div(
                      dcc.Dropdown(
                            id='dropdown',
                            options=[
                                {'label': 'Pixels ', 'value': 'pixels'},
                                {'label': 'Tag-Managers ', 'value': 'tag_managers'},
                                {'label': 'Ad-Tech ', 'value': 'ad_tech'},
                                {'label': 'Ad-Count ', 'value': 'ad_count'},
                                {'label': 'Cart/Checkout ', 'value': 'cart_checkout'},
                                {'label': 'Price-Points ', 'value': 'price_points'},
                                {'label': 'ECommerce-Stack ', 'value': 'ecommerce-stack'},
                                {'label': 'Payments ', 'value': 'payments'},
                                {'label': 'Social-Media-Presence ', 'value': 'social_media_presence'},
                                {'label': 'Android/iTunes ', 'value': 'android_itunes'},
                                {'label': 'Leadgen-Forms ', 'value': 'leadgen-form'},
                            ],
                            multi=True,
                            value='',
                            style={'width': '100%'}),className="eleven columns"),
                  html.Div(html.Button(id='submit-button-2', type='submit', children='Submit',title='The run time is approximately 2-3 mins for set of 10-12 websites',style={'color':'white','backgroundColor':'#506784'}),
                            style={'float':'right'})],className='row'),
                  html.Div(dcc.Checklist(id='select-all',options=[{'label': 'Select All', 'value': 1}], values=[],className='two columns'), id='checklist-container')

                    ],style={'marginBottom':20}
                ),

        html.Br(),  

        html.Div([html.P('Filter by Text Analytics:'),
                        html.Div([html.Div(
                            dcc.Dropdown(
                            id='dropdown-2',
                            options=[
                                {'label': 'Topic Modelling ', 'value': 'topic_modelling'},
                                {'label': 'Google API ', 'value': 'google_api'},
                                {'label': 'Automated Archetype ', 'value': 'automated_archetype'},
                            ],
                            multi=True,
                            value='',
                            style={'width': '100%'}),className="eleven columns"),
                        html.Div(html.Button(id='submit-button-3', type='submit', title='This process will take time for set of websites as we need to extract topics using NLP and ML based model. The run time is approximately 15 mins for set of 10-12 websites but for a single input website, this will finish in a minute.', children='Submit',style={'color':'white','backgroundColor':'#506784'}),
                                 style={'float':'right'})],className='row'),
                    ],
                ),
        
        html.Br(),

        html.Div([html.P('Filter by Image Analytics:'),
                        html.Div([html.Div(
                            dcc.Dropdown(
                            id='dropdown-3',
                            options=[
                                {'label': 'Image Detection ', 'value': 'image_detection'},
                                {'label': 'Image Recognition ', 'value': 'image_recognition'},
                            ],
                            multi=True,
                            value='',
                            style={'width': '100%'}),className="eleven columns"),
                        html.Div(html.Button(id='submit-button-4', type='submit', title='Image detection and recognition process is built on high memory consumption models so please think twice before clicking if you want to do it for list of websites. The run time is approximately 15 mins for set of 10-12 websites but for a single input website, this will finish in a minute or two.', children='Submit',style={'color':'white','backgroundColor':'#506784'}),
                                 style={'float':'right'})],className='row'),
                    ],
                ),
        
        html.Br(),

        html.Div([
                html.Div(id='download-link-1'),
                html.Div(html.A('Download Results',download="webCrawledData.csv",href="",target="_blank"),style={'display': 'none'}),], className="row"),

        html.Div([
                html.Div(id='table-header-1'),
                html.Div(html.P("Pixels/Ad-Techs/Tag-Managers/Other", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns",style={'display': 'none'}),
                html.Div(id='table-header-2'),
                html.Div(html.P("ECommerce/App/Lead-Gen-Forms", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns",style={'display': 'none'}),
                ],className="row"),

        html.Div([
                # html.Link(rel='stylesheet', href='/static/dash-datatable-light-dark.css'),
                # html.Div([html.P("Pixels/Ad-Techs/Tag-Managers/Other", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-2', className="six columns"),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),
                # html.Br(),
                # html.Div([html.P("ECommerce/App/Lead-Gen-Forms", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-3', className="six columns"),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),], className="row"),
        html.Div([html.Br()]),
                # html.Br(),
        
        html.Div([
                html.Div(id='table-header-3'),
                html.Div(html.P("Social-Media-Presence", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns",style={'display': 'none'}),
                html.Div(id='table-header-4'),
                html.Div(html.P("Channel-Assignment", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns",style={'display': 'none'}),
                ],className="row"),

        html.Div([
                # html.Div([html.P("Social-Media-Presence", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-4', className="six columns"),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),
                # html.Br(),
                # html.Div([html.P("Channel-Assignment", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-5', className="six columns"),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),], className="row"),
                            # html.Br(),

        html.Br(),

        html.Div([
                html.Div(id='download-link-2'),
                html.Div(html.A('Download Results',download="textCrawledData.csv",href="",target="_blank"),style={'display': 'none'}),], className="row"),

        html.Div([
                html.Div(id='table-header-5'),
                html.Div(html.P("Google-API-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns",style={'display': 'none'}),
                html.Div(id='table-header-6'),
                html.Div(html.P("Topic-Modelling-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns",style={'display': 'none'}),
                ],className="row"),

        html.Div([
                # html.Link(rel='stylesheet', href='/static/dash-datatable-light-dark.css'),
                
                # html.Div([html.P("Google-API-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-6', className="six columns"),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),
                # html.Br(),
                # html.Div([html.P("Topic-Modelling-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-7', className="six columns"),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),], className="row"),
                            # html.Br(),
        html.Div([html.Br()]),

        html.Div([
                html.Div(id='table-header-7'),
                html.Div(html.P("Final-Automated-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns",style={'display': 'none'}),
                ],className="row"),

        html.Div([
                # html.Div([html.P("Final-Automated-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-8', className="twelve columns"),
                html.Div(dt.DataTable(data=[{}],
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),], className="row"),

        html.Br(),

        html.Div([
                html.Div(id='download-link-3'),
                html.Div(html.A('Download Results',download="imageCrawledData.csv",href="",target="_blank"),style={'display': 'none'}),
                ], className="row"),

        html.Div([
                html.Div(id='table-header-8'),
                html.Div(html.P("Image-Analytics", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns",style={'display': 'none'}),
                ],className="row"),

        html.Div([
                # html.Div([html.P("Final-Automated-Archetype", className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold'})],style={'display': 'none'},className="six columns"),
                html.Div(id='datatable-9', className="twelve columns"),
                html.Div(dt.DataTable(data=[{}],
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),], className="row")

                            # html.Br(),
            ],
            className='row'
        ),

    dcc.Tab(label='Lead Search Engine Page', children=[
        html.Div([html.P('Upload List of Websites ("HTTP://WWW.XYZ.COM"):'),
                dcc.Upload(
                id='upload-data-file-L1',
                children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File')]),
                style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
                },
                    # Do not Allow multiple files to be uploaded
                    multiple=False
                ),
                # html.Div(id='output_div_1'),
                html.Div(id='datatable-L1'),
                html.Div(dt.DataTable(data=[{}],style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white'
                },filtering=True,
                        sorting=True,
                        sorting_type="multi",
                        row_selectable="multi",
                        selected_rows=[],
                        pagination_mode="fe",
                        pagination_settings={
                            "displayed_pages": 1,
                            "current_page": 0,
                            "page_size": 35,
                        },
                        navigation="page",), style={'display': 'none'}),
                html.Div(html.Br()),
                html.Div(html.Button(id='submit-button-L1', type='submit', children='Build Customized Lead Search Engine',style={'width':'100%','color':'white','backgroundColor':'#506784'}),
                         style={'float':'center'}),
                html.Div(html.Br()),
                html.Div(id='lead_div_1'),
                html.Div(dcc.Input(
                id = 'input_button_L1',
                placeholder='Enter query to search across web pages...',
                type='text',
                value='',
                style={'width': '100%'}
                ),className="eleven columns",style={'display': 'none'}),
                html.Div(id='lead_div_2'),
                html.Div(html.Button(id='submit-button-L2', type='submit', children='Submit',style={'color':'white','backgroundColor':'#506784'}),
                         style={'float':'right','display': 'none'}),
                ],style={'margin-top': '20'}),
        html.Br(),
        ],
        className='row'),

]),

    html.Link(href="https://use.fontawesome.com/releases/v5.2.0/css/all.css",rel="stylesheet"),
##        html.Link(href="https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css",rel="stylesheet"),
    html.Link(href="https://fonts.googleapis.com/css?family=Dosis", rel="stylesheet"),
    html.Link(href="https://fonts.googleapis.com/css?family=Open+Sans", rel="stylesheet"),
    html.Link(href="https://fonts.googleapis.com/css?family=Ubuntu", rel="stylesheet"),
    html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet")                     

],style={'marginLeft':20,'marginRight':20})

# login_layout = html.Div([
#             # represents the URL bar, doesn't render anything
#         dcc.Location(id='url', refresh=False),
#         html.Div(id='page-content'),
#         dcc.Input(id = 'user_name', placeholder='username', name='username', type='text'),
#         html.Br(),
#         dcc.Input(id = 'pass_word', placeholder='password', name='password', type='password'),
#         html.Br(),
#         html.Button(id='login_button', type='submit', children='Submit'),
#         # htmlx.Br(),
#         html.Div(id='output_login')
#         ])

# def serve_layout(login_success):
#     if(login_success==0):
#         return html.Div([
#             # represents the URL bar, doesn't render anything
#         dcc.Location(id='url', refresh=False),
#         html.Div(id='page-content'),
#         dcc.Input(id = 'user_name', placeholder='username', name='username', type='text'),
#         html.Br(),
#         dcc.Input(id = 'pass_word', placeholder='password', name='password', type='password'),
#         html.Br(),
#         html.Button(id='login_button', type='submit', children='Submit'),
#         # html.Br(),
#         html.Div(id='output_login')
#         ])
#     else:
#         return tab3_layout

login_layout = html.Div([
            html.P("Welcome to Web Crawler", className='title',style={'fontSize':25,'color':'black', 'fontWeight':'bold',"textAlign":"center", }),
            html.P("Please enter your credentials",style={'fontSize':15,'color':'black', 'fontWeight':'italic',"textAlign":"center", }),
            html.Div([
            dcc.Input(id = 'user_name', placeholder='Enter Username', name='username', type='text'),
        html.Br(),
        #html.Br(),
        dcc.Input(id = 'pass_word', placeholder='Enter Password', name='password', type='password'),
        html.Br(),html.Br(),
        html.Button(id='login_button', type='submit', children='Submit',style={'color':'white','fontWeight':'bold','backgroundColor':'black'}),
        html.Br(),html.Br(),
        html.Div(id='output_login'),html.Br(),
        html.Div(html.Img(src=app.get_asset_url('download.jfif'),
                          height="100%"),style={"float":"center"})],style={'margin':150, 'textAlign': 'center',"float":"center"}),],
                        style={'backgroundColor':'grey',"float":"center",'marginBottom':4,'marginTop':"4","borderBottom": "4px solid #000000",
                               "borderTop": "4px solid #000000","borderRight": "4px solid #000000","borderLeft": "4px solid #000000"})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(id='page-content2')
])

# def serve_layout(login_success):
#     if(login_success==0):
#         return html.Div([
#             html.P("Welcome to Web Crawler", className='title',style={'fontSize':25,'color':'black', 'fontWeight':'bold',"textAlign":"center", }),
#             html.P("Please enter your credentials",style={'fontSize':15,'color':'black', 'fontWeight':'italic',"textAlign":"center", }),
#             html.Div([
#             dcc.Input(id = 'user_name', placeholder='Enter Username', name='username', type='text'),
#         html.Br(),
#         # html.Br(),
#         dcc.Input(id = 'pass_word', placeholder='Enter Password', name='password', type='password'),
#         html.Br(),html.Br(),
#         html.Button(id='login_button', type='submit', children='Submit',style={'color':'black','fontWeight':'bold','backgroundColor':'green'}),
#         html.Br(),html.Br(),
#         html.Div(id='output_login'),
#         html.Div(html.Img(src='https://www.accenture.com/t20180719T114224Z__w__/in-en/_acnmedia/Accenture/DigitasLBi/new-applied-now/images/grid-items/Articles9/Accenture-main-logo.jpg',
#                           height="50%"),style={"float":"center"})],style={'margin':90, 'textAlign': 'center',"float":"center"}),
#         dcc.Location(id='url', refresh=False),
#         html.Div(id='page-content'),],
#                         style={'backgroundColor':'grey',"float":"center",'marginBottom':"4",'marginTop':"4","borderBottom": "4px solid #000000",
#                                "borderTop": "4px solid #000000","borderRight": "4px solid #000000","borderLeft": "4px solid #000000"})
#     else:
#         return tab3_layout

# app.layout = serve_layout(login_success)

# Index Page callback

@app.callback(Output('output_login', 'children'),
              [Input('login_button', 'n_clicks'),
              Input('url','pathname')],
              [State('user_name','value'),
               State('pass_word', 'value')])
def crawler_app_page(n_clicks,pathname,user_name, pass_word):

    if(n_clicks is not None):
        if(user_name!='atulag'):
        # if not username or not password:
            return html.P("Incorrect Username",style={'fontSize':15,'color':'darkred',"textAlign":"center" })
        elif(pass_word!='worldofcrawling'):
        # if not username or not password:
            return html.P("Incorrect Password",style={'fontSize':15,'color':'darkred',"textAlign":"center" })
        else:
            login_success=1
            user_url = '/web-crawler/app/logged-in'
            return dcc.Link(html.Button(id='login_button', type='login', children='login',
                                                  style={'color':'black','fontWeight':'bold','backgroundColor':'green'}),href=user_url)


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')],)
def display_page(pathname):
    if pathname == '/web-crawler/app/logged-in':
        return tab3_layout
    elif pathname == '/':
        return login_layout
    else:
        return '404: Error Not Found'

@app.callback(
    Output('dropdown', 'value'),
    [Input('select-all', 'values')],
    [State('dropdown', 'options')])
def test(selected, options):
    if len(selected) > 0:
        return [i['value'] for i in options]
    raise PreventUpdate()

@app.callback(
    Output('checklist-container', 'children'),
    [Input('dropdown', 'value')],
    [State('dropdown', 'options'),
     State('select-all', 'values')])
def tester(selected, options_1, checked):

    if len(selected) < len(options_1) and len(checked) == 0:
        raise PreventUpdate()

    elif len(selected) < len(options_1) and len(checked) == 1:
        return  dcc.Checklist(id='select-all',
                    options=[{'label': 'Select All', 'value': 1}], values=[])

    elif len(selected) == len(options_1) and len(checked) == 1:
        raise PreventUpdate()

    return  dcc.Checklist(id='select-all',
                    options=[{'label': 'Select All', 'value': 1}], values=[1])
# Loading screen CSS
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})


def pixelScore(df):
    if(df["No Pixel Flag"]==1):
        Pixel_Score = df["Criteo Pixel"]*2.4 + df["Facebook Pixel"]*1.4 + df["Pinterest Pixel"]*2.4 + df["Snap Pixel"]*6.0 + df["Twitter Pixel"]*1.6 + 6*df["Adobe Tag Manager Inbound"] +  6*df["Bright Tag Manager"] + 6*df["Google Tag Manager Inbound"] + 6*df["Tealium Manager"]
    else:
        Pixel_Score = df["Criteo Pixel"]*2.4 + df["Facebook Pixel"]*1.4 + df["Pinterest Pixel"]*2.4 + df["Snap Pixel"]*6.0 + df["Twitter Pixel"]*1.6 + 3*df["Adobe Tag Manager Inbound"] + 3*df["Bright Tag Manager"] + 3*df["Google Tag Manager Inbound"] + 3*df["Tealium Manager"]
    return Pixel_Score

def paymentScore(df):
    payment_score = df["Amazon Pay Flag"] + df["Apple Pay Flag"] + df["Chase Pay Flag"] + df["Google Pay Flag"] + df["Jcb Pay Flag"] + df["Masterpass Pay Flag"] + df["Paypal Pay Flag"] + df["Sage Pay Flag"] + df["Stripe Pay Flag"] + df["Visa Pay Flag"] + df["Mastercard Pay Flag"] + df["Discovery Pay Flag"] + df["Amex Pay Flag"] + df["Shopify Pay Flag"]
    return payment_score

def adTechScore(df):
    Ad_Tech_Score = df["Ad Roll Inbound"]*1.5 + df["Adobe Tag Manager Inbound"]*1.5 + df["Fb Exchange Inbound"]*1.5 + df["Google Ad Sense Inbound"]*1.5 + df["Google Remarketing Inbound"]*1.5 + df["Omniture Inbound"]*1.5 + df["Perfect Audience Inbound"]*1.5 + df["Wildfire Inbound"]*1.5
    return Ad_Tech_Score

def yelpScore(df):
    Yelp_Score = df["Yelp Badge"]*(-1)
    return Yelp_Score

def twitterER(df):
    twitter_er = 0
    if(df["Twitter Page"]==1):
        # print(df["Twitter_Likes"])
        if(df["Twitter Followers"]*df['Twitter Tweets']!=0):
            twitter_er = 100*(df["Twitter Likes"]+df["Twitter Post Replies"]+df["Twitter Post Retweets"])/(df["Twitter Followers"]*df['Twitter Tweets'])
    return twitter_er

def instaER(df):
    insta_er = 0
    if(df["Insta Page"]==1):
        if(df["Insta Followers"]*df['Insta Posts']!=0):
            insta_er = 100*(df["Insta Hashtag Posts"]*10)/(df["Insta Followers"]*df['Insta Posts'])
    return insta_er

# def ecommChannel(df):
#     if((df["R Dso Managed New"]==0)&(df["Fortune 1000"]==0)&(df["Inc 5000"]==0)&(df["R Public Client Domain"]==0)):
#         print(df[["R Dso Managed New","Fortune 1000","Inc 5000","R Public Client Domain"]])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df

def textCrawler(url):

    # Initializing dataframe
    df=pd.DataFrame()
    non_english_flag = 'en'
    text = ""
    cleaned_text = ""
    

    base=pd.read_csv('blackListed.csv' ,encoding ='utf-8')
    url_blacklist=[]
    for url_new in base['website']:
        url_blacklist.append(url_new)
    
    print (url)
    
    if ('FACEBOOK' in url) or ('LINKEDIN' in url) or ('GMAIL' in url) or ('YAHOO' in url) or ('INSTAGRAM' in url):
        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([text],columns=['Text']),pd.DataFrame([cleaned_text],columns=['Cleaned Text']),pd.DataFrame([non_english_flag],columns=['Non English Flag'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    if url in  ['HTTP://WWW.ALOHA6.COM','HTTP://WWW.FAZ.DE','HTTP://WWW.ALOHA6.COM','HTTP://WWW.WIRED.DE','HTTP://WWW.PRIMALLIFEORGANICS.COM','HTTP://WWW.ANDALUSIA-DENTALCENTERS.COM',
            'HTTP://WWW.SCIENCE37.COM','HTTP://WWW.ZOBELLO.COM','HTTP://WWW.AXISLABS.COM','HTTP://WWW.3POINT14.COM','HTTP://WWW.YUUPEE.COM',
            'HTTP://WWW.WBCOLL.EDU','HTTP://WWW.XENITH.COM','HTTP://WWW.BEAUFORCONGRESS.COM','HTTP://WWW.SUBARUSIXSTAR.COM','HTTP://WWW.YCOS.COM',
             'HTTP://WWW.BEIAMODA.COM','HTTP://BEAUFORCONGRESS.COM','HTTPS://WWW.YONKERSKIA.COM/?GCLID=EAIAIQOBCHMIUBJN5LD03AIVCFGNCH3EAG0LEAAYASAAEGJKTFD_BWE',
             'HTTP://WWW.4WDSUPACENTRE.COM.AU','HTTPS://WWW.THEYESITE.COM','HTTP://WWW.HUERLIMANNCC.COM','HTTP://WWW.RRMSPMC.COM',
             'HTTPS://CROSSFORTHENATIONS.ORG','HTTP://WWW.THEABCAFE.COM','HTTPS://WWW.WSWBA.COM','HTTP://WWW.TOUCHURBAN.COM']:

        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([text],columns=['Text']),pd.DataFrame([cleaned_text],columns=['Cleaned Text']),pd.DataFrame([non_english_flag],columns=['Non English Flag'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    if url in url_blacklist:
        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([text],columns=['Text']),pd.DataFrame([cleaned_text],columns=['Cleaned Text']),pd.DataFrame([non_english_flag],columns=['Non English Flag'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    try:     

        # Selenium Latest Web Driver crawling
        # chrome_options = Options()
        # chrome_options.add_argument('--dns-prefetch-disable')
        # chrome_options.add_argument('--no-sandbox')
        # # chrome_options.add_argument('--lang=en-US')
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument("--disable-logging")
        # chrome_options.add_argument('log-level=3')
        # # chrome_options.add_argument('--disable-gpu')
        # # chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en-US'})
        # browser = webdriver.Chrome(r"C:\\Users\\a.daluka\\Documents\\driver\\chromedriver.exe", chrome_options=chrome_options)
        # browser.get(url)
        # soup = BeautifulSoup(browser.page_source, 'lxml')
        # browser.quit()

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
        req = requests.get(url,headers=headers,timeout=100)
        req.raise_for_status()
        soup = BeautifulSoup(req.content, 'lxml')

        for script in soup(["script", "style", "meta", "noscript"]):
            script.extract()

        webText = soup.get_text()

        webT = re.sub("\s+" , " ", webText)
        # print(webT)
        text = webT

        text = text.replace('\n', ' ').replace('\r', ' ').replace('\r\n', ' ')

        try:
            print(detect(text))
            text = str(text)
            if detect(text) != 'en':
                print("Non English Language detected")
                non_english_flag = detect(text)
                for k in text.split("\n"):
                    final = " ".join(re.findall(r"[a-zA-Z]+", k))
                    cleaned_text = cleaned_text+" "+final
                cleaned_text = cleaned_text.strip()
                cleaned_text = re.sub(' +', ' ',cleaned_text)
                cleaned_text = cleaned_text.lower()
            else:
                for k in text.split("\n"):
                #     print(re.sub(r"[^a-zA-Z0-9]+", ' ', k))
                #     Or:
                    final = " ".join(re.findall(r"[a-zA-Z]+", k))
                    cleaned_text = cleaned_text+" "+final
                cleaned_text = cleaned_text.strip()
                cleaned_text = re.sub(' +', ' ',cleaned_text)
                cleaned_text = cleaned_text.lower()
        except Exception:
            exception_flag = 1
            try:
                non_english_flag = detect(text)
            except Exception:
                non_english_flag = 'en'
            text = str(text)
            for k in text.split("\n"):
                final = " ".join(re.findall(r"[a-zA-Z]+", k))
                cleaned_text = cleaned_text+" "+final
            cleaned_text = cleaned_text.strip()
            cleaned_text = re.sub(' +', ' ',cleaned_text)
            cleaned_text = cleaned_text.lower()

        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([text],columns=['Text']),pd.DataFrame([cleaned_text],columns=['Cleaned Text']),pd.DataFrame([non_english_flag],columns=['Non English Flag'])],axis=1)
        df=df.append(temp)

    except Exception:
        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([text],columns=['Text']),pd.DataFrame([cleaned_text],columns=['Cleaned Text']),pd.DataFrame([non_english_flag],columns=['Non English Flag'])],axis=1)
        df=df.append(temp)

    return(str(df.values.tolist()))

def classify_update(text, verbose=True):
    """Classify the input text into categories. """
    
    # Initializing dataframe
    df=pd.DataFrame()

    google_api_confidence = ''
    google_api_sub_category = ''
    google_api_category = ''
    google_api_archetype = ''

    # print(type(text))
    if(type(text)==float and np.isnan(text)):
        temp=pd.concat([pd.DataFrame([''],columns=['Cleaned Text']),pd.DataFrame([google_api_confidence],columns=['Google Api Confidence']),pd.DataFrame([google_api_sub_category],columns=['Google Api Sub Category']),pd.DataFrame([google_api_category],columns=['Google Api Category']),pd.DataFrame([google_api_archetype],columns=['Google Api Archetype'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    try:

        language_client = language.LanguageServiceClient()
        document = language.types.Document(
            content=text,
            type=language.enums.Document.Type.PLAIN_TEXT)
        response = language_client.classify_text(document)
        
        categories = response.categories

        num_categories = len(categories)

        if(num_categories==0):
            google_api_confidence = ''
            google_api_sub_category = ''
            google_api_category = ''
            google_api_archetype = ''
        else:
            google_api_confidence = categories[0].confidence
            google_api_sub_category = categories[0].name
            print(google_api_sub_category,google_api_confidence)

            if('Adult' in google_api_sub_category):
                google_api_category = 'Adult'
                google_api_archetype = 'Blacklist'
            elif('Arts & Entertainment' in google_api_sub_category):
                google_api_category = 'Art & Entertainment'
                google_api_archetype = 'Lead Gen'
            elif('Autos & Vehicles' in google_api_sub_category):
                google_api_category = 'Autos & Vehicles'
                google_api_archetype = 'Lead Gen'
            elif('Beauty & Fitness' in google_api_sub_category):
                google_api_category = 'Beauty & Fitness'
                google_api_archetype = 'Lead Gen'
            elif('Books & Literature' in google_api_sub_category):
                google_api_category = 'Books & Literature'
                google_api_archetype = 'Lead Gen'
            elif('Business & Industrial' in google_api_sub_category):
                business_agency_list = ['/Business & Industrial/Advertising & Marketing/Public Relations','/Business & Industrial/Business Operations',
                '/Business & Industrial/Business Operations/Business Plans & Presentations','/Business & Industrial/Business Operations/Management',
                '/Business & Industrial/Business Services','/Business & Industrial/Business Services/Consulting']
                if any(x in google_api_sub_category for x in business_agency_list):
                    google_api_category = 'Business Services'
                    google_api_archetype = 'Agency'
                else:
                    if('/Business & Industrial/Business Services/E-Commerce Services' in google_api_sub_category):
                        google_api_category = 'Business & Industrial'
                        google_api_archetype = 'Lead Gen'
                    else:
                        google_api_category = 'Business Services'
                        google_api_archetype = 'Lead Gen'
            elif('Computers & Electronics' in google_api_sub_category):
                google_api_category = 'Computers & Electronics'
                comp_ecommerce_list = ['Computer Hardware','Consumer Electronics','/Computers & Electronics/Electronics & Electrical/Electronic Components',
                '/Computers & Electronics/Electronics & Electrical/Power Supplies']
                if any(x in google_api_sub_category for x in comp_ecommerce_list):
                    google_api_archetype = 'eCommerce'
                else:
                    google_api_archetype = 'Lead Gen'
            elif('/Finance' in google_api_sub_category):
                google_api_category = 'Finance'
                google_api_archetype = 'Lead Gen'
            elif('Food & Drink' in google_api_sub_category):
                google_api_category = 'Food & Drinks'
                google_api_archetype = 'Other'
            elif('Games' in google_api_sub_category):
                google_api_category = 'Computers & Electronics'
                games_lead_list = ['Board Games','Card Games','Family-Oriented Games & Activities','Table Games','Word Games',
                'Gambling','/Games/Puzzles & Brainteasers','/Games/Roleplaying Games']
                if any(x in google_api_sub_category for x in games_lead_list):
                    google_api_archetype = 'App Dev'
                else:
                    google_api_archetype = 'Lead Gen'
            elif('/Health' in google_api_sub_category):
                google_api_category = 'Health'
                google_api_archetype = 'Lead Gen'
            elif('/Hobbies & Leisure' in google_api_sub_category):
                google_api_category = 'Hobbies & Leisure'
                google_api_archetype = 'Lead Gen'
            elif('/Home & Garden' in google_api_sub_category):
                google_api_category = 'Home & Garden'
                google_api_archetype = 'Lead Gen'
            elif('/Internet & Telecom' in google_api_sub_category):
                google_api_category = 'Internet & Telecom'
                internet_app_list = ['/Internet & Telecom/Mobile & Wireless/Mobile Apps & Add-Ons','Email & Messaging']
                if any(x in google_api_sub_category for x in internet_app_list):
                    google_api_archetype = 'App Dev'
                elif('/Internet & Telecom/Mobile & Wireless/Mobile Phones' in google_api_sub_category):
                    google_api_archetype = 'eCommerce'
                else:
                    google_api_archetype = 'Lead Gen'
            elif('/Jobs & Education' in google_api_sub_category):
                google_api_category = 'Jobs & Education'
                google_api_archetype = 'Lead Gen'
            elif('/Law & Government' in google_api_sub_category):
                google_api_category = 'Law & Government'
                government_list = ['/Law & Government/Government','/Law & Government/Military','/Law & Government/Public Safety/Law Enforcement']
                if any(x in google_api_sub_category for x in government_list):
                    google_api_archetype = 'Government'
                else:
                    google_api_archetype = 'Lead Gen'
            elif('/News' in google_api_sub_category):
                google_api_category = 'News'
                google_api_archetype = 'Lead Gen'
            elif('/Online Communities' in google_api_sub_category):
                google_api_category = 'Online Communities'
                oc_app_list = ['/Online Communities/Online Goodies/Social Network Apps & Add-Ons','/Online Communities/Social Networks',
                '/Online Communities/Virtual Worlds']
                if any(x in google_api_sub_category for x in oc_app_list):
                    google_api_archetype = 'App Dev'
                else:
                    google_api_archetype = 'Lead Gen'
            elif('/People & Society' in google_api_sub_category):
                google_api_category = 'Communities'
                google_api_archetype = 'Lead Gen'
            elif('/Pets & Animals' in google_api_sub_category):
                google_api_category = 'Pets & Animals'
                google_api_archetype = 'Lead Gen'
            elif('/Real Estate' in google_api_sub_category):
                google_api_category = 'Real Estate'
                google_api_archetype = 'Lead Gen'
            elif('/Reference' in google_api_sub_category):
                google_api_category = 'Reference & Science'
                google_api_archetype = 'Lead Gen'
            elif('/Science' in google_api_sub_category):
                google_api_category = 'Reference & Science'
                google_api_archetype = 'Lead Gen'
            elif('/Sensitive Subjects' in google_api_sub_category):
                google_api_category = 'Sensitive Subject'
                google_api_archetype = 'Blacklist'
            elif('/Shopping' in google_api_sub_category):
                if('/Shopping/Tobacco Products' in google_api_sub_category):
                    google_api_category = 'Tobacco'
                    google_api_archetype = 'Blacklist'
                else:
                    google_api_archetype = 'eCommerce'
                    if('/Shopping/Apparel/' in google_api_sub_category):
                        google_api_category = 'Fashion & Accessories'
                    elif('/Shopping/Toys' in google_api_sub_category):
                        google_api_category = 'Toys & Hobbies'
                    elif('/Shopping/Luxury Goods' in google_api_sub_category):
                        google_api_category = 'Fashion & Accessories'
                    elif('/Shopping/Gifts & Special Event Items' in google_api_sub_category):
                        google_api_category = 'Gifts'
                    elif('/Shopping/Antiques & Collectibles' in google_api_sub_category):
                        google_api_category = 'Gifts'
                    else:
                        google_api_category = 'Shopping'
            elif('/Sports' in google_api_sub_category):
                google_api_category = 'Sports'
                google_api_archetype = 'Lead Gen'
            elif('/Travel' in google_api_sub_category):
                google_api_category = 'Travel'
                google_api_archetype = 'Lead Gen'
            else:
                google_api_category = ''
                google_api_archetype = ''
        # for category in categories:
        #     # Turn the categories into a dictionary of the form:
        #     # {category.name: category.confidence}, so that they can
        #     # be treated as a sparse vector.
        #     try:
        #         result[category.name] = category.confidence
        #     except:
        #         return {'No prediction':0}
        # if verbose:
    #     #print(text)
    #     for category in categories:
    #         try:
    #             print(u'=' * 20)
    #             print(u'{:<16}: {}'.format('category', category.name))
    #             print(u'{:<16}: {}'.format('confidence', category.confidence))
    #         except:
    #             return {'No prediction':0}

        temp=pd.concat([pd.DataFrame([text],columns=['Cleaned Text']),pd.DataFrame([google_api_confidence],columns=['Google Api Confidence']),pd.DataFrame([google_api_sub_category],columns=['Google Api Sub Category']),pd.DataFrame([google_api_category],columns=['Google Api Category']),pd.DataFrame([google_api_archetype],columns=['Google Api Archetype'])],axis=1)
        df=df.append(temp)

    except Exception:
        temp=pd.concat([pd.DataFrame([text],columns=['Cleaned Text']),pd.DataFrame([google_api_confidence],columns=['Google Api Confidence']),pd.DataFrame([google_api_sub_category],columns=['Google Api Sub Category']),pd.DataFrame([google_api_category],columns=['Google Api Category']),pd.DataFrame([google_api_archetype],columns=['Google Api Archetype'])],axis=1)
        df=df.append(temp)

    # print(df)
    # print(str(df.values.tolist()))
    return(str(df.values.tolist()))
    

def topic_archetype(df):

  topic_modelling_category = ''
  topic_modelling_archetype = ''

  if('en' not in df['Non English Flag'] and df['Non English Flag'] != ''):
    topic_modelling_category = 'International'
    topic_modelling_archetype = 'International'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  if('.GOV' in df['Website']):
    topic_modelling_category = 'Government'
    topic_modelling_archetype = 'Government'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  if('.EDU' in df['Website']):
    topic_modelling_category = 'Jobs & Education'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  # print(df['Keywords'])
  
  word_list = df['Keywords'].split(', ')

  print(word_list)

  adult = ['porn','ass','sex','adult','fuck','marijuana','tobacco','milf']

  in_adult = len(list(set(adult) & set(word_list)))

  if(in_adult>=3):
    topic_modelling_category = 'Adult'
    topic_modelling_archetype = 'Blacklist'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  games = ['Game','battle','video','play','tournament','mobile','android','itunes','click','pubg','app']

  in_games = len(list(set(games) & set(word_list)))

  if(in_games>=4):
    topic_modelling_category = 'Games'
    topic_modelling_archetype = 'App Dev'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  application = ['app','itunes','android','play_store','app_store','mobile','message','chat','video','voice']

  in_application = len(list(set(application) & set(word_list)))

  if(in_application>=4):
    topic_modelling_category = 'Application'
    topic_modelling_archetype = 'App Dev'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  communities = ['people','love','experience','life','talk','story','audience','connect','message','family','friends','society','relationship']

  in_communities = len(list(set(communities) & set(word_list)))

  if(in_communities>=4):
    topic_modelling_category = 'Communities'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  news = ['news','global','world','business','International','public','issue']

  in_news = len(list(set(news) & set(word_list)))

  if(in_news>=3):
    topic_modelling_category = 'News'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  online_communities = ['online','contact','blog','email','facebook','instagram','twitter','newsletter','subscribe','login','sign','community']

  in_online_communities = len(list(set(online_communities) & set(word_list)))

  if(in_online_communities>=4):
    topic_modelling_category = 'Online Communities'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  invalid = ['invalid','website','site','not','domain','sale','sell','privacy']

  in_invalid = len(list(set(invalid) & set(word_list)))

  if(in_invalid>=3):
    topic_modelling_category = 'Invalid'
    topic_modelling_archetype = 'Invalid'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  business_leadgen = ['business','service','digital','strategy','product','technology','design','learn','education','finance','industry']

  in_business_leadgen = len(list(set(business_leadgen) & set(word_list)))

  if(in_business_leadgen>=4):
    topic_modelling_category = 'Business & Industrial'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  business_agency = ['agency','advertise','advertising','campaign','digital','marketing','consulting','brand','management','design','client','solution']

  in_business_agency = len(list(set(business_agency) & set(word_list)))

  if(in_business_agency>=4):
    topic_modelling_category = 'Business & Industrial'
    topic_modelling_archetype = 'Agency'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  health = ['care','patient','medical','health','treatement','hospital','dental','appointment','clinic']

  in_health = len(list(set(health) & set(word_list)))

  if(in_health>=4):
    topic_modelling_category = 'Health'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  travel = ['city','travel','park','hotel','island','beach','tour','flight','trip','ticket','museum','resort']

  in_travel = len(list(set(travel) & set(word_list)))

  if(in_travel>=4):
    topic_modelling_category = 'Travel'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  shopping = ['shop','product','sale','cart','price','collection','dress','color','gift','accessories','shirt','skin','body','review','checkout','woman','women','fashion','toy']

  in_shopping = len(list(set(shopping) & set(word_list)))

  if(in_shopping>=4):
    topic_modelling_category = 'Shopping'
    topic_modelling_archetype = 'eCommerce'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  comp_electroonics = ['computer','harddisk','iphone','mobile','tablet','samsung','stylus','laptop','gadget','camera','touch','android','apple','mac','chrome']

  in_comp_electroonics = len(list(set(comp_electroonics) & set(word_list)))

  if(in_comp_electroonics>=4):
    topic_modelling_category = 'Computer & Electronics'
    topic_modelling_archetype = 'eCommerce'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  food = ['food','menu','restaurant','serve','beer','burger','wine','hotel','chocolate','coffee','pizza','recipe','sauce','drink','breakfast','lunch','dinner','brunch','cheese','chicken']

  in_food = len(list(set(food) & set(word_list)))

  if(in_food>=4):
    topic_modelling_category = 'Food & Drinks'
    topic_modelling_archetype = 'Other'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  business_agency = ['music','event','food','party','live','night','film','game','club','ticket','entertainment','friday']

  in_business_agency = len(list(set(business_agency) & set(word_list)))

  if(in_business_agency>=4):
    topic_modelling_category = 'Arts & Entertainment'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  real_estate = ['property','estate','sale','home','house','apartment','dubai','villa','residence','real_estate']

  in_real_estate = len(list(set(real_estate) & set(word_list)))

  if(in_real_estate>=3):
    topic_modelling_category = 'Real Estate'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  finance = ['finance','insurance','business','account','loan','rate','credit','card','bank','banking']

  in_finance = len(list(set(finance) & set(word_list)))

  if(in_finance>=4):
    topic_modelling_category = 'Finance'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  sp_fitness = ['sport','team','club','training','fitness','game','schedule','coach','gym','yoga','gymnassium','exercise','crossfit']

  in_sp_fitness = len(list(set(sp_fitness) & set(word_list)))

  if(in_sp_fitness>=4):
    topic_modelling_category = 'Sports & Fitness'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  auto = ['auto','vehicle','car','model','bicycle','license','drive','driving']

  in_auto = len(list(set(auto) & set(word_list)))

  if(in_auto>=2):
    topic_modelling_category = 'Autos & Vehicles'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  education = ['education','student','school','program','university','learn','college','campus','study','class','academic','admission','board','career']

  in_education = len(list(set(education) & set(word_list)))

  if(in_education>=4):
    topic_modelling_category = 'Jobs & Education'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  hobbies = ['hobby','leisure','hobbies','club','craft','outdoor']

  in_hobbies = len(list(set(hobbies) & set(word_list)))

  if(in_hobbies>=3):
    topic_modelling_category = 'Hobbies & Leisure'
    topic_modelling_archetype = 'Lead Gen'
    df["Topic Modelling Category"] = topic_modelling_category
    df["Topic Modelling Archetype"] = topic_modelling_archetype
    return df

  df["Topic Modelling Category"] = topic_modelling_category
  df["Topic Modelling Archetype"] = topic_modelling_archetype
  return df

def auto_archetype(df):
  if(type(df['Google Api Archetype'])==float and np.isnan(df['Google Api Archetype'])):
    df['Google Api Archetype']=''
  print((df['Topic Modelling Archetype']),(df['Google Api Archetype']))
  if(df['Topic Modelling Archetype']!='' and df['Google Api Archetype']==''):
    automated_archetype = df['Topic Modelling Archetype']
  elif(df['Topic Modelling Archetype']=='' and df['Google Api Archetype']!=''):
    automated_archetype = df['Google Api Archetype']
  elif(df['Topic Modelling Archetype']=='' and df['Google Api Archetype']==''):
    automated_archetype = 'No archetype'
  else:
    if(df['Topic Modelling Archetype']==df['Google Api Archetype']):
      automated_archetype = df['Google Api Archetype']
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='Lead Gen'):
      automated_archetype = 'eCommerce'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='Lead Gen'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='Lead Gen'):
      automated_archetype = 'Lead Gen'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='Lead Gen'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='Lead Gen'):
      automated_archetype = 'App Dev'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='Lead Gen'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Other' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'Other'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'App Dev'
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'eCommerce'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'Agency'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='App Dev'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'Other'
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'Other'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'Other'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'Other'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='Other'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Other' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'Agency'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'Agency'
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'Agency'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'Agency'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='Agency'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Other' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'Other'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'eCommerce'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'Agency'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'eCommerce'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='eCommerce'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Other' and df['Google Api Archetype']=='International'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='International'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='International'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='International'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='International'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='International'):
      automated_archetype = 'International'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='International'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Other' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Government' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='Blacklist'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='Other' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='Lead Gen' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='Agency' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='eCommerce' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='App Dev' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'Government'
    elif(df['Topic Modelling Archetype']=='Blacklist' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'Blacklist'
    elif(df['Topic Modelling Archetype']=='International' and df['Google Api Archetype']=='Government'):
      automated_archetype = 'International'
    else:
      automated_archetype = ''

  return automated_archetype

def Content_type(url):

    # Initializing dataframe
    df=pd.DataFrame()

    crawled_flag_website = 0
    shopify_flag = 0
    check_flag = 0

    snap_pixel = 0
    pinterest_pixel = 0
    facebook_pixel = 0
    twitter_pixel = 0
    criteo_pixel = 0

    google_ad_sense_inbound = 0
    google_remarketing_inbound = 0

    fb_exchange_inbound = 0
    ad_roll_inbound = 0
    perfect_audience_inbound = 0
    wildfire_inbound = 0
    omniture_inbound = 0

    google_tag_manager_inbound = 0
    adobe_tag_manager_inbound = 0

    google_ad_sense_outbound = 0
    google_remarketing_outbound = 0

    fb_exchange_outbound = 0
    ad_roll_outbound = 0
    perfect_audience_outbound = 0
    wildfire_outbound = 0
    omniture_outbound = 0

    google_tag_manager_outbound = 0
    adobe_tag_manager_outbound = 0

    bright_tag_manager = 0
    tealium_manager = 0
    tagman_manager = 0

    android_flag = 0
    android_link  = ''
    itunes_flag = 0

    snapchat_badge = 0
    pinterest_badge = 0
    facebook_badge = 0
    instagram_badge = 0
    twitter_badge = 0
    linkedin_badge = 0
    yelp_badge = 0
    youtube_badge = 0
    google_badge = 0

    cart_hopping = 0
    checkout_hopping = 0

    magento_flag = 0
    paypalpay_flag = 0
    amazon_flag = 0
    bigcommerce_flag = 0
    squarespace_flag = 0

    mastercardpay_flag = 0
    visapay_flag = 0
    amexpay_flag = 0
    applepay_flag = 0
    googlepay_flag = 0
    shopifypay_flag = 0
    masterpasspay_flag = 0
    amazonpay_flag = 0
    stripepay_flag = 0
    chasepay_flag = 0
    discoverypay_flag = 0
    jcbpay_flag = 0
    sagepay_flag = 0

    dig_google_tag_manager_inbound = 0
    gtm_hopping = 0

    google_ad_services_flag = 0
    yahoo_ad_services_flag = 0
    aol_ad_services_flag = 0
    bing_ad_services_flag = 0
    amazon_ad_services_flag = 0

    google_ad_count = 0
    yahoo_ad_count = 0
    aol_ad_count = 0
    bing_ad_count = 0
    amazon_ad_count = 0

    price_count = 0
    currencySymbol = ''
    productString = ''
    min_price = ''
    max_price = ''
    avg_price = ''
    rupeeList = ['rs','Rs','RS','₹','INR']
    dollarList = ['$','USD']
    euroList = ['€','EUR']
    CADList = ['C$','CAD']
    poundList = ['£']

    Followers = ''
    Following = ''
    Likes = ''
    Tweets = ''
    Post_Replies = ''
    Post_Retweets = ''
    Post_Likes = ''

    hash_posts = ''
    Insta_Followers = ''
    Insta_Following = ''
    Insta_Posts = ''
    Insta_first_post_likes = ''

    itunes_app_flag = 0
    itunes_developer_flag = 0
    app_id = ''
    itunes_url = ''
    app_title = ''
    app_subtitle = ''
    app_identity = ''
    app_ranking = ''
    app_price = ''
    app_purchase = ''
    app_description = ''
    app_rating = ''
    app_rating_count = ''
    app_seller = ''
    app_size = ''
    app_category = ''
    app_age_rating = ''

    Leadgen_form = 0

    ECom_Subarch_Subscription = 0
    ECom_Subarch_PetFood = 0
    ECom_Subarch_FastFashion = 0
    ECom_Subarch_JewelryAndAccessories = 0
    ECom_Subarch_CustomizedGifts = 0
    ECom_Subarch_HomeGoods = 0
    ECom_Subarch_Technology = 0
    ECom_Subarch_Travel = 0
    ECom_Subarch_Sustainable = 0
    ECom_Subarch_Beauty = 0
    ECom_Subarch_FestivalAndMusicEvents = 0
    ECom_Subarch_VideoGames = 0
    ECom_Subarch_MoviesAndEntertainment = 0
    ECom_Subarch_SportsAndFitness = 0
    ECom_Subarch_ToysAndHobbies = 0

    base=pd.read_csv('blackListed.csv' ,encoding ='utf-8')
    url_blacklist=[]
    for url_new in base['website']:
        url_blacklist.append(url_new)
    
    print (url)
    
    if ('FACEBOOK' in url) or ('LINKEDIN' in url) or ('GMAIL' in url) or ('YAHOO' in url) or ('INSTAGRAM' in url):
        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([crawled_flag_website],columns=['crawled_flag_website']),pd.DataFrame([shopify_flag],columns=['shopify_flag']),
            pd.DataFrame([android_flag],columns=['android_flag']),pd.DataFrame([android_link],columns=['android_link']),pd.DataFrame([itunes_flag],columns=['itunes_flag']),
            pd.DataFrame([cart_hopping],columns=['cart_hopping']),pd.DataFrame([checkout_hopping],columns=['checkout_hopping']),pd.DataFrame([Leadgen_form],columns=['Leadgen_form']),
            pd.DataFrame([ECom_Subarch_Subscription],columns=["ECom_Subarch_Subscription"]),pd.DataFrame([ECom_Subarch_PetFood],columns=["ECom_Subarch_PetFood"]),pd.DataFrame([ECom_Subarch_FastFashion],columns=["ECom_Subarch_FastFashion"]),pd.DataFrame([ECom_Subarch_JewelryAndAccessories],columns=["ECom_Subarch_JewelryAndAccessories"]),pd.DataFrame([ECom_Subarch_CustomizedGifts],columns=["ECom_Subarch_CustomizedGifts"]),pd.DataFrame([ECom_Subarch_HomeGoods],columns=["ECom_Subarch_HomeGoods"]),pd.DataFrame([ECom_Subarch_Technology],columns=["ECom_Subarch_Technology"]),pd.DataFrame([ECom_Subarch_Travel],columns=["ECom_Subarch_Travel"]),pd.DataFrame([ECom_Subarch_Sustainable],columns=["ECom_Subarch_Sustainable"]),pd.DataFrame([ECom_Subarch_Beauty],columns=["ECom_Subarch_Beauty"]),pd.DataFrame([ECom_Subarch_FestivalAndMusicEvents],columns=["ECom_Subarch_FestivalAndMusicEvents"]),pd.DataFrame([ECom_Subarch_VideoGames],columns=["ECom_Subarch_VideoGames"]),pd.DataFrame([ECom_Subarch_MoviesAndEntertainment],columns=["ECom_Subarch_MoviesAndEntertainment"]),pd.DataFrame([ECom_Subarch_SportsAndFitness],columns=["ECom_Subarch_SportsAndFitness"]),pd.DataFrame([ECom_Subarch_ToysAndHobbies],columns=["ECom_Subarch_ToysAndHobbies"]),
            pd.DataFrame([magento_flag],columns=['magento_flag']),pd.DataFrame([paypalpay_flag],columns=['paypalpay_flag']),pd.DataFrame([amazon_flag],columns=['amazon_flag']),pd.DataFrame([bigcommerce_flag],columns=['bigcommerce_flag']),pd.DataFrame([squarespace_flag],columns=['squarespace_flag']),
            pd.DataFrame([mastercardpay_flag],columns=['mastercardpay_flag']),pd.DataFrame([visapay_flag],columns=['visapay_flag']),pd.DataFrame([amexpay_flag],columns=['amexpay_flag']),pd.DataFrame([applepay_flag],columns=['applepay_flag']),pd.DataFrame([googlepay_flag],columns=['googlepay_flag']),
            pd.DataFrame([shopifypay_flag],columns=['shopifypay_flag']),pd.DataFrame([masterpasspay_flag],columns=['masterpasspay_flag']),pd.DataFrame([amazonpay_flag],columns=['amazonpay_flag']),pd.DataFrame([stripepay_flag],columns=['stripepay_flag']),
            pd.DataFrame([chasepay_flag],columns=['chasepay_flag']),pd.DataFrame([discoverypay_flag],columns=['discoverypay_flag']),pd.DataFrame([jcbpay_flag],columns=['jcbpay_flag']),pd.DataFrame([sagepay_flag],columns=['sagepay_flag']),
            pd.DataFrame([snap_pixel],columns=['snap_pixel']),pd.DataFrame([pinterest_pixel],columns=['pinterest_pixel']),pd.DataFrame([facebook_pixel],columns=['facebook_pixel']),pd.DataFrame([twitter_pixel],columns=['twitter_pixel']),pd.DataFrame([criteo_pixel],columns=['criteo_pixel']),pd.DataFrame([google_ad_sense_inbound],columns=['google_ad_sense_inbound']),pd.DataFrame([google_remarketing_inbound],columns=['google_remarketing_inbound']),
            pd.DataFrame([fb_exchange_inbound],columns=['fb_exchange_inbound']),pd.DataFrame([ad_roll_inbound],columns=['ad_roll_inbound']),pd.DataFrame([perfect_audience_inbound],columns=['perfect_audience_inbound']),pd.DataFrame([wildfire_inbound],columns=['wildfire_inbound']),pd.DataFrame([omniture_inbound],columns=['omniture_inbound']),
            pd.DataFrame([google_tag_manager_inbound],columns=['google_tag_manager_inbound']),pd.DataFrame([adobe_tag_manager_inbound],columns=['adobe_tag_manager_inbound']),
            pd.DataFrame([google_ad_sense_outbound],columns=['google_ad_sense_outbound']),pd.DataFrame([google_remarketing_outbound],columns=['google_remarketing_outbound']),
            pd.DataFrame([fb_exchange_outbound],columns=['fb_exchange_outbound']),pd.DataFrame([ad_roll_outbound],columns=['ad_roll_outbound']),pd.DataFrame([perfect_audience_outbound],columns=['perfect_audience_outbound']),pd.DataFrame([wildfire_outbound],columns=['wildfire_outbound']),pd.DataFrame([omniture_outbound],columns=['omniture_outbound']),
            pd.DataFrame([google_tag_manager_outbound],columns=['google_tag_manager_outbound']),pd.DataFrame([adobe_tag_manager_outbound],columns=['adobe_tag_manager_outbound']),
            pd.DataFrame([bright_tag_manager],columns=['bright_tag_manager']),pd.DataFrame([tealium_manager],columns=['tealium_manager']),pd.DataFrame([tagman_manager],columns=['tagman_manager']),
            pd.DataFrame([snapchat_badge],columns=['snapchat_badge']),pd.DataFrame([pinterest_badge],columns=['pinterest_badge']),pd.DataFrame([facebook_badge],columns=['facebook_badge']),pd.DataFrame([instagram_badge],columns=['instagram_badge']),pd.DataFrame([twitter_badge],columns=['twitter_badge']),pd.DataFrame([linkedin_badge],columns=['linkedin_badge']),pd.DataFrame([yelp_badge],columns=['yelp_badge']),pd.DataFrame([youtube_badge],columns=['youtube_badge']),pd.DataFrame([google_badge],columns=['google_badge']),
            pd.DataFrame([google_ad_services_flag],columns=['google_ad_services_flag']),pd.DataFrame([google_ad_count],columns=['google_ad_count']),pd.DataFrame([yahoo_ad_services_flag],columns=['yahoo_ad_services_flag']),pd.DataFrame([yahoo_ad_count],columns=['yahoo_ad_count']),pd.DataFrame([aol_ad_services_flag],columns=['aol_ad_services_flag']),
            pd.DataFrame([aol_ad_count],columns=['aol_ad_count']),pd.DataFrame([bing_ad_services_flag],columns=['bing_ad_services_flag']),pd.DataFrame([bing_ad_count],columns=['bing_ad_count']),pd.DataFrame([amazon_ad_services_flag],columns=['amazon_ad_services_flag']),pd.DataFrame([amazon_ad_count],columns=['amazon_ad_count']),
            pd.DataFrame([price_count],columns=['price_count']),pd.DataFrame([productString],columns=['product_categories']),pd.DataFrame([currencySymbol],columns=['currency_symbol']),pd.DataFrame([min_price],columns=['min_price']),pd.DataFrame([max_price],columns=['max_price']),pd.DataFrame([avg_price],columns=['avg_price']),
            pd.DataFrame([Followers],columns=['Twitter_Followers']),pd.DataFrame([Following],columns=['Twitter_Following']),pd.DataFrame([Likes],columns=['Twitter_Likes']),pd.DataFrame([Tweets],columns=['Twitter_Tweets']),pd.DataFrame([Post_Replies],columns=['Twitter_Post_Replies']),pd.DataFrame([Post_Retweets],columns=['Twitter_Post_Retweets']),pd.DataFrame([Post_Likes],columns=['Twitter_Post_Likes']),
            pd.DataFrame([Insta_Followers],columns=['Insta_Followers']),pd.DataFrame([Insta_Following],columns=['Insta_Following']),pd.DataFrame([Insta_Posts],columns=['Insta_Posts']),pd.DataFrame([hash_posts],columns=['Insta_Hashtag_Posts']),
            pd.DataFrame([itunes_url],columns=['itunes_app_link']),pd.DataFrame([itunes_app_flag],columns=['itunes_app_flag']),pd.DataFrame([itunes_developer_flag],columns=['itunes_developer_flag']),pd.DataFrame([app_id],columns=['itunes_app_id']),pd.DataFrame([app_title],columns=['itunes_app_title']),pd.DataFrame([app_subtitle],columns=['itunes_app_subtitle']),pd.DataFrame([app_identity],columns=['itunes_app_identity']),pd.DataFrame([app_ranking],columns=['itunes_app_ranking']),pd.DataFrame([app_price],columns=['itunes_app_price']),
            pd.DataFrame([app_purchase],columns=['itunes_app_purchase']),pd.DataFrame([app_description],columns=['itunes_app_description']),pd.DataFrame([app_rating],columns=['itunes_app_rating']),pd.DataFrame([app_rating_count],columns=['itunes_app_rating_count']),pd.DataFrame([app_seller],columns=['itunes_app_seller']),pd.DataFrame([app_size],columns=['itunes_app_size']),pd.DataFrame([app_category],columns=['itunes_app_category']),pd.DataFrame([app_age_rating],columns=['itunes_app_age_rating'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    if url in  ['HTTP://WWW.ALOHA6.COM','HTTP://WWW.FAZ.DE','HTTP://WWW.ALOHA6.COM','HTTP://WWW.WIRED.DE','HTTP://WWW.PRIMALLIFEORGANICS.COM','HTTP://WWW.ANDALUSIA-DENTALCENTERS.COM',
            'HTTP://WWW.SCIENCE37.COM','HTTP://WWW.ZOBELLO.COM','HTTP://WWW.AXISLABS.COM','HTTP://WWW.3POINT14.COM','HTTP://WWW.YUUPEE.COM',
            'HTTP://WWW.WBCOLL.EDU','HTTP://WWW.XENITH.COM','HTTP://WWW.BEAUFORCONGRESS.COM','HTTP://WWW.SUBARUSIXSTAR.COM','HTTP://WWW.YCOS.COM',
             'HTTP://WWW.BEIAMODA.COM','HTTP://BEAUFORCONGRESS.COM','HTTPS://WWW.YONKERSKIA.COM/?GCLID=EAIAIQOBCHMIUBJN5LD03AIVCFGNCH3EAG0LEAAYASAAEGJKTFD_BWE',
             'HTTP://WWW.4WDSUPACENTRE.COM.AU','HTTPS://WWW.THEYESITE.COM','HTTP://WWW.HUERLIMANNCC.COM','HTTP://WWW.RRMSPMC.COM',
             'HTTPS://CROSSFORTHENATIONS.ORG','HTTP://WWW.THEABCAFE.COM','HTTPS://WWW.WSWBA.COM','HTTP://WWW.TOUCHURBAN.COM']:

        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([crawled_flag_website],columns=['crawled_flag_website']),pd.DataFrame([shopify_flag],columns=['shopify_flag']),
            pd.DataFrame([android_flag],columns=['android_flag']),pd.DataFrame([android_link],columns=['android_link']),pd.DataFrame([itunes_flag],columns=['itunes_flag']),
            pd.DataFrame([cart_hopping],columns=['cart_hopping']),pd.DataFrame([checkout_hopping],columns=['checkout_hopping']),pd.DataFrame([Leadgen_form],columns=['Leadgen_form']),
            pd.DataFrame([ECom_Subarch_Subscription],columns=["ECom_Subarch_Subscription"]),pd.DataFrame([ECom_Subarch_PetFood],columns=["ECom_Subarch_PetFood"]),pd.DataFrame([ECom_Subarch_FastFashion],columns=["ECom_Subarch_FastFashion"]),pd.DataFrame([ECom_Subarch_JewelryAndAccessories],columns=["ECom_Subarch_JewelryAndAccessories"]),pd.DataFrame([ECom_Subarch_CustomizedGifts],columns=["ECom_Subarch_CustomizedGifts"]),pd.DataFrame([ECom_Subarch_HomeGoods],columns=["ECom_Subarch_HomeGoods"]),pd.DataFrame([ECom_Subarch_Technology],columns=["ECom_Subarch_Technology"]),pd.DataFrame([ECom_Subarch_Travel],columns=["ECom_Subarch_Travel"]),pd.DataFrame([ECom_Subarch_Sustainable],columns=["ECom_Subarch_Sustainable"]),pd.DataFrame([ECom_Subarch_Beauty],columns=["ECom_Subarch_Beauty"]),pd.DataFrame([ECom_Subarch_FestivalAndMusicEvents],columns=["ECom_Subarch_FestivalAndMusicEvents"]),pd.DataFrame([ECom_Subarch_VideoGames],columns=["ECom_Subarch_VideoGames"]),pd.DataFrame([ECom_Subarch_MoviesAndEntertainment],columns=["ECom_Subarch_MoviesAndEntertainment"]),pd.DataFrame([ECom_Subarch_SportsAndFitness],columns=["ECom_Subarch_SportsAndFitness"]),pd.DataFrame([ECom_Subarch_ToysAndHobbies],columns=["ECom_Subarch_ToysAndHobbies"]),
            pd.DataFrame([magento_flag],columns=['magento_flag']),pd.DataFrame([paypalpay_flag],columns=['paypalpay_flag']),pd.DataFrame([amazon_flag],columns=['amazon_flag']),pd.DataFrame([bigcommerce_flag],columns=['bigcommerce_flag']),pd.DataFrame([squarespace_flag],columns=['squarespace_flag']),
            pd.DataFrame([mastercardpay_flag],columns=['mastercardpay_flag']),pd.DataFrame([visapay_flag],columns=['visapay_flag']),pd.DataFrame([amexpay_flag],columns=['amexpay_flag']),pd.DataFrame([applepay_flag],columns=['applepay_flag']),pd.DataFrame([googlepay_flag],columns=['googlepay_flag']),
            pd.DataFrame([shopifypay_flag],columns=['shopifypay_flag']),pd.DataFrame([masterpasspay_flag],columns=['masterpasspay_flag']),pd.DataFrame([amazonpay_flag],columns=['amazonpay_flag']),pd.DataFrame([stripepay_flag],columns=['stripepay_flag']),
            pd.DataFrame([chasepay_flag],columns=['chasepay_flag']),pd.DataFrame([discoverypay_flag],columns=['discoverypay_flag']),pd.DataFrame([jcbpay_flag],columns=['jcbpay_flag']),pd.DataFrame([sagepay_flag],columns=['sagepay_flag']),
            pd.DataFrame([snap_pixel],columns=['snap_pixel']),pd.DataFrame([pinterest_pixel],columns=['pinterest_pixel']),pd.DataFrame([facebook_pixel],columns=['facebook_pixel']),pd.DataFrame([twitter_pixel],columns=['twitter_pixel']),pd.DataFrame([criteo_pixel],columns=['criteo_pixel']),pd.DataFrame([google_ad_sense_inbound],columns=['google_ad_sense_inbound']),pd.DataFrame([google_remarketing_inbound],columns=['google_remarketing_inbound']),
            pd.DataFrame([fb_exchange_inbound],columns=['fb_exchange_inbound']),pd.DataFrame([ad_roll_inbound],columns=['ad_roll_inbound']),pd.DataFrame([perfect_audience_inbound],columns=['perfect_audience_inbound']),pd.DataFrame([wildfire_inbound],columns=['wildfire_inbound']),pd.DataFrame([omniture_inbound],columns=['omniture_inbound']),
            pd.DataFrame([google_tag_manager_inbound],columns=['google_tag_manager_inbound']),pd.DataFrame([adobe_tag_manager_inbound],columns=['adobe_tag_manager_inbound']),
            pd.DataFrame([google_ad_sense_outbound],columns=['google_ad_sense_outbound']),pd.DataFrame([google_remarketing_outbound],columns=['google_remarketing_outbound']),
            pd.DataFrame([fb_exchange_outbound],columns=['fb_exchange_outbound']),pd.DataFrame([ad_roll_outbound],columns=['ad_roll_outbound']),pd.DataFrame([perfect_audience_outbound],columns=['perfect_audience_outbound']),pd.DataFrame([wildfire_outbound],columns=['wildfire_outbound']),pd.DataFrame([omniture_outbound],columns=['omniture_outbound']),
            pd.DataFrame([google_tag_manager_outbound],columns=['google_tag_manager_outbound']),pd.DataFrame([adobe_tag_manager_outbound],columns=['adobe_tag_manager_outbound']),pd.DataFrame([bright_tag_manager],columns=['bright_tag_manager']),pd.DataFrame([tealium_manager],columns=['tealium_manager']),pd.DataFrame([tagman_manager],columns=['tagman_manager']),
            pd.DataFrame([snapchat_badge],columns=['snapchat_badge']),pd.DataFrame([pinterest_badge],columns=['pinterest_badge']),pd.DataFrame([facebook_badge],columns=['facebook_badge']),pd.DataFrame([instagram_badge],columns=['instagram_badge']),pd.DataFrame([twitter_badge],columns=['twitter_badge']),pd.DataFrame([linkedin_badge],columns=['linkedin_badge']),pd.DataFrame([yelp_badge],columns=['yelp_badge']),pd.DataFrame([youtube_badge],columns=['youtube_badge']),pd.DataFrame([google_badge],columns=['google_badge']),
            pd.DataFrame([google_ad_services_flag],columns=['google_ad_services_flag']),pd.DataFrame([google_ad_count],columns=['google_ad_count']),pd.DataFrame([yahoo_ad_services_flag],columns=['yahoo_ad_services_flag']),pd.DataFrame([yahoo_ad_count],columns=['yahoo_ad_count']),pd.DataFrame([aol_ad_services_flag],columns=['aol_ad_services_flag']),
            pd.DataFrame([aol_ad_count],columns=['aol_ad_count']),pd.DataFrame([bing_ad_services_flag],columns=['bing_ad_services_flag']),pd.DataFrame([bing_ad_count],columns=['bing_ad_count']),pd.DataFrame([amazon_ad_services_flag],columns=['amazon_ad_services_flag']),pd.DataFrame([amazon_ad_count],columns=['amazon_ad_count']),
            pd.DataFrame([price_count],columns=['price_count']),pd.DataFrame([productString],columns=['product_categories']),pd.DataFrame([currencySymbol],columns=['currency_symbol']),pd.DataFrame([min_price],columns=['min_price']),pd.DataFrame([max_price],columns=['max_price']),pd.DataFrame([avg_price],columns=['avg_price']),
            pd.DataFrame([Followers],columns=['Twitter_Followers']),pd.DataFrame([Following],columns=['Twitter_Following']),pd.DataFrame([Likes],columns=['Twitter_Likes']),pd.DataFrame([Tweets],columns=['Twitter_Tweets']),pd.DataFrame([Post_Replies],columns=['Twitter_Post_Replies']),pd.DataFrame([Post_Retweets],columns=['Twitter_Post_Retweets']),pd.DataFrame([Post_Likes],columns=['Twitter_Post_Likes']),
            pd.DataFrame([Insta_Followers],columns=['Insta_Followers']),pd.DataFrame([Insta_Following],columns=['Insta_Following']),pd.DataFrame([Insta_Posts],columns=['Insta_Posts']),pd.DataFrame([hash_posts],columns=['Insta_Hashtag_Posts']),
            pd.DataFrame([itunes_url],columns=['itunes_app_link']),pd.DataFrame([itunes_app_flag],columns=['itunes_app_flag']),pd.DataFrame([itunes_developer_flag],columns=['itunes_developer_flag']),pd.DataFrame([app_id],columns=['itunes_app_id']),pd.DataFrame([app_title],columns=['itunes_app_title']),pd.DataFrame([app_subtitle],columns=['itunes_app_subtitle']),pd.DataFrame([app_identity],columns=['itunes_app_identity']),pd.DataFrame([app_ranking],columns=['itunes_app_ranking']),pd.DataFrame([app_price],columns=['itunes_app_price']),
            pd.DataFrame([app_purchase],columns=['itunes_app_purchase']),pd.DataFrame([app_description],columns=['itunes_app_description']),pd.DataFrame([app_rating],columns=['itunes_app_rating']),pd.DataFrame([app_rating_count],columns=['itunes_app_rating_count']),pd.DataFrame([app_seller],columns=['itunes_app_seller']),pd.DataFrame([app_size],columns=['itunes_app_size']),pd.DataFrame([app_category],columns=['itunes_app_category']),pd.DataFrame([app_age_rating],columns=['itunes_app_age_rating'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    if url in url_blacklist:
        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([crawled_flag_website],columns=['crawled_flag_website']),pd.DataFrame([shopify_flag],columns=['shopify_flag']),
            pd.DataFrame([android_flag],columns=['android_flag']),pd.DataFrame([android_link],columns=['android_link']),pd.DataFrame([itunes_flag],columns=['itunes_flag']),
            pd.DataFrame([cart_hopping],columns=['cart_hopping']),pd.DataFrame([checkout_hopping],columns=['checkout_hopping']),pd.DataFrame([Leadgen_form],columns=['Leadgen_form']),
            pd.DataFrame([ECom_Subarch_Subscription],columns=["ECom_Subarch_Subscription"]),pd.DataFrame([ECom_Subarch_PetFood],columns=["ECom_Subarch_PetFood"]),pd.DataFrame([ECom_Subarch_FastFashion],columns=["ECom_Subarch_FastFashion"]),pd.DataFrame([ECom_Subarch_JewelryAndAccessories],columns=["ECom_Subarch_JewelryAndAccessories"]),pd.DataFrame([ECom_Subarch_CustomizedGifts],columns=["ECom_Subarch_CustomizedGifts"]),pd.DataFrame([ECom_Subarch_HomeGoods],columns=["ECom_Subarch_HomeGoods"]),pd.DataFrame([ECom_Subarch_Technology],columns=["ECom_Subarch_Technology"]),pd.DataFrame([ECom_Subarch_Travel],columns=["ECom_Subarch_Travel"]),pd.DataFrame([ECom_Subarch_Sustainable],columns=["ECom_Subarch_Sustainable"]),pd.DataFrame([ECom_Subarch_Beauty],columns=["ECom_Subarch_Beauty"]),pd.DataFrame([ECom_Subarch_FestivalAndMusicEvents],columns=["ECom_Subarch_FestivalAndMusicEvents"]),pd.DataFrame([ECom_Subarch_VideoGames],columns=["ECom_Subarch_VideoGames"]),pd.DataFrame([ECom_Subarch_MoviesAndEntertainment],columns=["ECom_Subarch_MoviesAndEntertainment"]),pd.DataFrame([ECom_Subarch_SportsAndFitness],columns=["ECom_Subarch_SportsAndFitness"]),pd.DataFrame([ECom_Subarch_ToysAndHobbies],columns=["ECom_Subarch_ToysAndHobbies"]),
            pd.DataFrame([magento_flag],columns=['magento_flag']),pd.DataFrame([paypalpay_flag],columns=['paypalpay_flag']),pd.DataFrame([amazon_flag],columns=['amazon_flag']),pd.DataFrame([bigcommerce_flag],columns=['bigcommerce_flag']),pd.DataFrame([squarespace_flag],columns=['squarespace_flag']),
            pd.DataFrame([mastercardpay_flag],columns=['mastercardpay_flag']),pd.DataFrame([visapay_flag],columns=['visapay_flag']),pd.DataFrame([amexpay_flag],columns=['amexpay_flag']),pd.DataFrame([applepay_flag],columns=['applepay_flag']),pd.DataFrame([googlepay_flag],columns=['googlepay_flag']),
            pd.DataFrame([shopifypay_flag],columns=['shopifypay_flag']),pd.DataFrame([masterpasspay_flag],columns=['masterpasspay_flag']),pd.DataFrame([amazonpay_flag],columns=['amazonpay_flag']),pd.DataFrame([stripepay_flag],columns=['stripepay_flag']),
            pd.DataFrame([chasepay_flag],columns=['chasepay_flag']),pd.DataFrame([discoverypay_flag],columns=['discoverypay_flag']),pd.DataFrame([jcbpay_flag],columns=['jcbpay_flag']),pd.DataFrame([sagepay_flag],columns=['sagepay_flag']),
            pd.DataFrame([snap_pixel],columns=['snap_pixel']),pd.DataFrame([pinterest_pixel],columns=['pinterest_pixel']),pd.DataFrame([facebook_pixel],columns=['facebook_pixel']),pd.DataFrame([twitter_pixel],columns=['twitter_pixel']),pd.DataFrame([criteo_pixel],columns=['criteo_pixel']),pd.DataFrame([google_ad_sense_inbound],columns=['google_ad_sense_inbound']),pd.DataFrame([google_remarketing_inbound],columns=['google_remarketing_inbound']),
            pd.DataFrame([fb_exchange_inbound],columns=['fb_exchange_inbound']),pd.DataFrame([ad_roll_inbound],columns=['ad_roll_inbound']),pd.DataFrame([perfect_audience_inbound],columns=['perfect_audience_inbound']),pd.DataFrame([wildfire_inbound],columns=['wildfire_inbound']),pd.DataFrame([omniture_inbound],columns=['omniture_inbound']), 
            pd.DataFrame([google_tag_manager_inbound],columns=['google_tag_manager_inbound']),pd.DataFrame([adobe_tag_manager_inbound],columns=['adobe_tag_manager_inbound']),
            pd.DataFrame([google_ad_sense_outbound],columns=['google_ad_sense_outbound']),pd.DataFrame([google_remarketing_outbound],columns=['google_remarketing_outbound']),
            pd.DataFrame([fb_exchange_outbound],columns=['fb_exchange_outbound']),pd.DataFrame([ad_roll_outbound],columns=['ad_roll_outbound']),pd.DataFrame([perfect_audience_outbound],columns=['perfect_audience_outbound']),pd.DataFrame([wildfire_outbound],columns=['wildfire_outbound']),pd.DataFrame([omniture_outbound],columns=['omniture_outbound']),
            pd.DataFrame([google_tag_manager_outbound],columns=['google_tag_manager_outbound']),pd.DataFrame([adobe_tag_manager_outbound],columns=['adobe_tag_manager_outbound']),pd.DataFrame([bright_tag_manager],columns=['bright_tag_manager']),pd.DataFrame([tealium_manager],columns=['tealium_manager']),pd.DataFrame([tagman_manager],columns=['tagman_manager']),
            pd.DataFrame([snapchat_badge],columns=['snapchat_badge']),pd.DataFrame([pinterest_badge],columns=['pinterest_badge']),pd.DataFrame([facebook_badge],columns=['facebook_badge']),pd.DataFrame([instagram_badge],columns=['instagram_badge']),pd.DataFrame([twitter_badge],columns=['twitter_badge']),pd.DataFrame([linkedin_badge],columns=['linkedin_badge']),pd.DataFrame([yelp_badge],columns=['yelp_badge']),pd.DataFrame([youtube_badge],columns=['youtube_badge']),pd.DataFrame([google_badge],columns=['google_badge']),
            pd.DataFrame([google_ad_services_flag],columns=['google_ad_services_flag']),pd.DataFrame([google_ad_count],columns=['google_ad_count']),pd.DataFrame([yahoo_ad_services_flag],columns=['yahoo_ad_services_flag']),pd.DataFrame([yahoo_ad_count],columns=['yahoo_ad_count']),pd.DataFrame([aol_ad_services_flag],columns=['aol_ad_services_flag']),
            pd.DataFrame([aol_ad_count],columns=['aol_ad_count']),pd.DataFrame([bing_ad_services_flag],columns=['bing_ad_services_flag']),pd.DataFrame([bing_ad_count],columns=['bing_ad_count']),pd.DataFrame([amazon_ad_services_flag],columns=['amazon_ad_services_flag']),pd.DataFrame([amazon_ad_count],columns=['amazon_ad_count']),
            pd.DataFrame([price_count],columns=['price_count']),pd.DataFrame([productString],columns=['product_categories']),pd.DataFrame([currencySymbol],columns=['currency_symbol']),pd.DataFrame([min_price],columns=['min_price']),pd.DataFrame([max_price],columns=['max_price']),pd.DataFrame([avg_price],columns=['avg_price']),
            pd.DataFrame([Followers],columns=['Twitter_Followers']),pd.DataFrame([Following],columns=['Twitter_Following']),pd.DataFrame([Likes],columns=['Twitter_Likes']),pd.DataFrame([Tweets],columns=['Twitter_Tweets']),pd.DataFrame([Post_Replies],columns=['Twitter_Post_Replies']),pd.DataFrame([Post_Retweets],columns=['Twitter_Post_Retweets']),pd.DataFrame([Post_Likes],columns=['Twitter_Post_Likes']),
            pd.DataFrame([Insta_Followers],columns=['Insta_Followers']),pd.DataFrame([Insta_Following],columns=['Insta_Following']),pd.DataFrame([Insta_Posts],columns=['Insta_Posts']),pd.DataFrame([hash_posts],columns=['Insta_Hashtag_Posts']),
            pd.DataFrame([itunes_url],columns=['itunes_app_link']),pd.DataFrame([itunes_app_flag],columns=['itunes_app_flag']),pd.DataFrame([itunes_developer_flag],columns=['itunes_developer_flag']),pd.DataFrame([app_id],columns=['itunes_app_id']),pd.DataFrame([app_title],columns=['itunes_app_title']),pd.DataFrame([app_subtitle],columns=['itunes_app_subtitle']),pd.DataFrame([app_identity],columns=['itunes_app_identity']),pd.DataFrame([app_ranking],columns=['itunes_app_ranking']),pd.DataFrame([app_price],columns=['itunes_app_price']),
            pd.DataFrame([app_purchase],columns=['itunes_app_purchase']),pd.DataFrame([app_description],columns=['itunes_app_description']),pd.DataFrame([app_rating],columns=['itunes_app_rating']),pd.DataFrame([app_rating_count],columns=['itunes_app_rating_count']),pd.DataFrame([app_seller],columns=['itunes_app_seller']),pd.DataFrame([app_size],columns=['itunes_app_size']),pd.DataFrame([app_category],columns=['itunes_app_category']),pd.DataFrame([app_age_rating],columns=['itunes_app_age_rating'])],axis=1)
        df=df.append(temp)
        return(str(df.values.tolist()))

    try:     

        # Selenium Latest Web Driver crawling
        # chrome_options = Options()
        # chrome_options.add_argument('--dns-prefetch-disable')
        # chrome_options.add_argument('--no-sandbox')
        # # chrome_options.add_argument('--lang=en-US')
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument("--disable-logging")
        # chrome_options.add_argument('log-level=3')
        # # chrome_options.add_argument('--disable-gpu')
        # # chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en-US'})
        # browser = webdriver.Chrome(r"C:\\Users\\a.daluka\\Documents\\driver\\chromedriver.exe", chrome_options=chrome_options)
        # browser.get(url)
        # soup = BeautifulSoup(browser.page_source, 'lxml')
        # browser.quit()

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
        req = requests.get(url,headers=headers,timeout=100)
        req.raise_for_status()
        soup = BeautifulSoup(req.content, 'lxml')

        crawled_flag_website = 1
        if("/CART" in url):
            cart_url = url
        else:
            cart_url = url+"/CART"

        if("/CHECKOUT" in url):
            checkout_url = url
        else:   
            checkout_url = url+"/CHECKOUT"

        cart_url = cart_url.lower()
        checkout_url = checkout_url.lower()
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
        try:
            cart_req = requests.get(cart_url,headers=headers,timeout=15)
            cart_req.raise_for_status()
            cart_hopping = 1
            print("cart_hopping",1)
        except Exception:
            pass
        try:
            checkout_req = requests.get(checkout_url,headers=headers,timeout=15)
            checkout_req.raise_for_status()
            checkout_hopping = 1
            print("checkout_hopping",1)
        except Exception:
            pass

        styles = soup.find_all('style')
        for styleElement in styles:
             # Payment Links Capturing
            if(paypalpay_flag==1):
                pass
            else:
                resultPaypalStyle = styleElement.find_all(string=re.compile('paypal',re.IGNORECASE), recursive=True)
                if resultPaypalStyle:
                    paypalpay_flag = 1
                    print("Paypal is there")
            if(mastercardpay_flag==1):
                pass
            else:
                resultMasterCardStyle = styleElement.find_all(string=re.compile('mastercard',re.IGNORECASE), recursive=True)
                if resultMasterCardStyle:
                    mastercardpay_flag = 1
                    print("Master Card is there")
            if(visapay_flag==1):
                pass
            else:
                resultVisaStyle = styleElement.find_all(string=re.compile('visa',re.IGNORECASE), recursive=True)
                if resultVisaStyle:
                    visapay_flag = 1
                    print("Visa is there")
            if(amexpay_flag==1):
                pass
            else:
                resultAmexStyle = styleElement.find_all(string=re.compile('amex',re.IGNORECASE), recursive=True)
                resultAmexStyle_2 = styleElement.find_all(string=re.compile('americanexpress',re.IGNORECASE), recursive=True)
                if resultAmexStyle:
                    amexpay_flag = 1
                    print("Amex is there")
                elif resultAmexStyle_2:
                    amexpay_flag = 1
                    print("Amex is there")
                else:
                    pass
            if(applepay_flag==1):
                pass
            else:
                resultApplePayStyle = styleElement.find_all(string=re.compile('apple-pay',re.IGNORECASE), recursive=True)
                if resultApplePayStyle:
                    applepay_flag = 1
                    print("Apple Pay is there")
            if(googlepay_flag==1):
                pass
            else:
                resultGooglePayStyle = styleElement.find_all(string=re.compile('google-pay',re.IGNORECASE), recursive=True)
                if resultGooglePayStyle:
                    googlepay_flag = 1
                    print("Google Pay is there")
            if(shopifypay_flag==1):
                pass
            else:
                resultShopifyPayStyle = styleElement.find_all(string=re.compile('shopify-pay',re.IGNORECASE), recursive=True)
                if resultShopifyPayStyle:
                    shopifypay_flag = 1
                    print("Shopify Pay is there")
            if(masterpasspay_flag==1):
                pass
            else:
                resultMasterPassPayStyle = styleElement.find_all(string=re.compile('masterpass',re.IGNORECASE), recursive=True)
                if resultMasterPassPayStyle:
                    masterpasspay_flag = 1
                    print("Masterpass Pay is there")
            if(amazonpay_flag==1):
                pass
            else:
                resultAmazonPayStyle = styleElement.find_all(string=re.compile('amazon-pay',re.IGNORECASE), recursive=True)
                if resultAmazonPayStyle:
                    amazonpay_flag = 1
                    print("Amazon Pay is there")
            # if(stripepay_flag==1):
            #     pass
            # else:
            #     resultStripePayStyle = styleElement.find_all(string=re.compile('stripe',re.IGNORECASE), recursive=True)
            #     if resultStripePayStyle:
            #         stripepay_flag = 1
            #         print("Stripe Pay is there")

        # scriptTags = soup.find_all('script',{"type":"text/javascript"})
        scripts = soup.find_all('script')
        # print(scripts)
        # print(scriptTags)
        # print(scripts)

        pixelList = ['snaptr','pintrk','fbq','twq','criteo_','adsbygoogle','gtag.js']
        for element in scripts:
            # print(element)
            if(snap_pixel==1):
                pass
            else:
                # search for snap pixel
                snapElements = ['snaptr']
                resultSnap = element.find_all(string=re.compile('snaptr',re.IGNORECASE), recursive=True)
                if resultSnap:
                    snap_pixel = 1
                    print("Snapchat pixel is there")
                else:
                    if 'src' in element.attrs:
                        snap_src = element.get('src')
                        snap_src_list=['tr.snapchat.com','scevent.min.js']
                        # print(snap_src)
                        if any(x in snap_src for x in snap_src_list):
                        # if 'scevent.min.js' in snap_src:
                            snap_pixel = 1
                            print("Snapchat pixel is there")
            if(pinterest_pixel==1):
                pass
            else:
                # search for pinterest pixel
                resultPinterest = element.find_all(string=re.compile('pintrk',re.IGNORECASE), recursive=True)
                if resultPinterest:
                    pinterest_pixel = 1
                    print("Pinterest pixel is there")
                else:
                    if 'src' in element.attrs:
                        pinterest_src = element.get('src')
                        # print(pinterest_src)
                        if 's.pinimg.com/ct/core.js' in pinterest_src:
                            pinterest_pixel = 1
                            print("Pinterest pixel is there")
            if(facebook_pixel==1):
                pass
            else:
                # search for facebook pixel
                # print(element)
                resultFacebook = element.find_all(string=re.compile('fbq', re.IGNORECASE),recursive=True)
                # print(resultFacebook)
                if resultFacebook:
                    if 'fbq(' in resultFacebook[0]:
                        facebook_pixel = 1
                        print("Facebook pixel is there")
                    elif '_fbq' in resultFacebook[0]:
                        facebook_pixel = 1
                        print("Facebook pixel is there")
                    else:
                        if 'src' in element.attrs:
                            facebook_src = element.get('src')
                            # print(facebook_src)
                            if 'connect.facebook.net/en_US/fbevents.js' in facebook_src:
                                facebook_pixel = 1
                                print("Facebook pixel is there")
                else:
                    if 'src' in element.attrs:
                        facebook_src = element.get('src')
                        # print(facebook_src)
                        if 'connect.facebook.net/en_US/fbevents.js' in facebook_src:
                            facebook_pixel = 1
                            print("Facebook pixel is there")
            if(twitter_pixel==1):
                pass
            else:
                # search for facebook pixel
                # print(element)
                resultTwitter = element.find_all(string=re.compile('twq', re.IGNORECASE),recursive=True)
                # print(resultFacebook)
                if resultTwitter:
                    if 'twq(' in resultTwitter[0]:
                        twitter_pixel = 1
                        print("Twitter pixel is there")
                    elif '_twq' in resultTwitter[0]:
                        twitter_pixel = 1
                        print("Twitter pixel is there")
                    else:
                        if 'src' in element.attrs:
                            twitter_src = element.get('src')
                            # print(twitter_src)
                            twitter_src_list = ['platform.twitter.com/oct.js','platform.twitter.com/widgets.js']
                            if any(x in twitter_src for x in twitter_src_list):
                            # if 'platform.twitter.com' in twitter_src:
                                twitter_pixel = 1
                                print("Twitter pixel is there")

                else:
                    if 'src' in element.attrs:
                        twitter_src = element.get('src')
                        # print(twitter_src)
                        twitter_src_list = ['platform.twitter.com/oct.js','platform.twitter.com/widgets.js']
                        if any(x in twitter_src for x in twitter_src_list):
                        # if 'platform.twitter.com' in twitter_src:
                            twitter_pixel = 1
                            print("Twitter pixel is there")

            if(criteo_pixel==1):
                pass
            else:
                resultCriteo = element.find_all(string=re.compile('criteo_',re.IGNORECASE), recursive=True)
                if resultCriteo:
                    criteo_pixel = 1
                    print("Criteo pixel is there")
                else:
                    if 'src' in element.attrs:
                        criteo_src = element.get('src')
                        if 'static.criteo.net' in criteo_src:
                            criteo_pixel = 1
                            print("Criteo pixel is there")
            
            if(google_ad_sense_inbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultGoogleAdSense = element.find_all(string=re.compile('googleadservices',re.IGNORECASE), recursive=True)
                if resultGoogleAdSense:
                    if 'conversion' in resultGoogleAdSense[0]:
                        google_ad_sense_inbound = 1
                        print("Google Ad Sense is there")
                else:
                    # print(element)
                    if 'src' in element.attrs:
                        googleadsense_src = element.get('src')
                        # print(adroll_src)
                        if 'googleadservices' in googleadsense_src:
                            if 'conversion' in googleadsense_src:
                                google_ad_sense_inbound = 1
                                print("Google Ad Sense is there")
            if(google_ad_sense_outbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultGoogleAdSenseOutbound = element.find_all(string=re.compile('pagead2.googlesyndication.com',re.IGNORECASE), recursive=True)
                resultGoogleAdSenseOutbound_2 = element.find_all(string=re.compile('enable_page_level_ads: true',re.IGNORECASE), recursive=True)
                if resultGoogleAdSenseOutbound:
                    google_ad_sense_outbound = 1
                    print("Google Ad Sense Outbound is there")
                elif resultGoogleAdSenseOutbound_2:
                    google_ad_sense_outbound = 1
                    print("Google Ad Sense Outbound is there")
                else:
                    # print(element)
                    if 'src' in element.attrs:
                        googleadsense_src_outbound = element.get('src')
                        # print(adroll_src)
                        if 'googleadservices' in googleadsense_src_outbound:
                            if 'conversion'  not in googleadsense_src_outbound:
                                google_ad_sense_outbound = 1
                                print("Google Ad Sense Outbound is there")

            if(google_remarketing_inbound==1):
                pass
            else:
                # search for google remarketing pixel
                resultGoogleRemarketing = element.find_all(string=re.compile('google_remarketing_inbound',re.IGNORECASE), recursive=True)
                if resultGoogleRemarketing:
                    google_remarketing_inbound = 1
                    print("Google Remarketing is there")
                else:
                    if 'src' in element.attrs:
                        googleremarketing_src = element.get('src')
                        # print(adroll_src)
                        if 'g.doubleclick' in googleremarketing_src:
                            if 'viewthroughconversion' in googleremarketing_src:
                                google_remarketing_inbound = 1
                                print("Google Remarketing is there")
                            else:
                                google_remarketing_outbound = 1
                                print("Google Remarketing Outbound is there")           

            if(fb_exchange_inbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultFbExchange = element.find_all(string=re.compile('fbq',re.IGNORECASE), recursive=True)
                resultFbExchange_2 = element.find_all(string=re.compile('fb_exchange_inbound_token',re.IGNORECASE), recursive=True)
                if resultFbExchange:
                    if 'fbq(' in resultFbExchange[0]:
                        fb_exchange_inbound = 1
                        print("Fb Exchange is there")
                if resultFbExchange_2:
                    fb_exchange_inbound = 1
                    print("Fb Exchange is there")

            if(ad_roll_inbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultAdRoll = element.find_all(string=re.compile('_adroll_loaded',re.IGNORECASE), recursive=True)
                if resultAdRoll:
                    ad_roll_inbound = 1
                    print("Ad Roll is there")
                else:
                    # print(element)
                    if 'src' in element.attrs:
                        adroll_src = element.get('src')
                        # print(adroll_src)
                        if 's.adroll.com' in adroll_src:
                            ad_roll_inbound = 1
                            print("Ad Roll is there")
            if(ad_roll_outbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultAdRollOutbound = element.find_all(string=re.compile('www.adroll.com',re.IGNORECASE), recursive=True)
                if resultAdRollOutbound:
                    ad_roll_outbound = 1
                    print("Ad Roll Outbound is there")
                else:
                    # print(element)
                    if 'src' in element.attrs:
                        adroll_src_outbound = element.get('src')
                        # print(adroll_src)
                        adroll_src_list=['d.adroll.com','app.adroll.com']
                        # print(snap_src)
                        if any(x in adroll_src_outbound for x in adroll_src_list):
                        # if 'd.adroll.com' in adroll_src_outbound:
                            ad_roll_outbound = 1
                            print("Ad Roll Outbound is there")

            if(perfect_audience_inbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultPerfectAudience = element.find_all(string=re.compile('window._pa',re.IGNORECASE), recursive=True)
                if resultPerfectAudience:
                    perfect_audience_inbound = 1
                    print("Perfect Audience is there")
            if(perfect_audience_outbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultPerfectAudienceOutbound = element.find_all(string=re.compile('app.perfectaudience.com',re.IGNORECASE), recursive=True)
                if resultPerfectAudienceOutbound:
                    perfect_audience_outbound = 1
                    print("Perfect Audience Outbound is there")

            if(wildfire_inbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultWildfire_inbound = element.find_all(string=re.compile('wildfire',re.IGNORECASE), recursive=True)
                if resultWildfire_inbound:
                    wildfire_inbound = 1
                    print("Wildfire_inbound is there")
                else:
                    if 'src' in element.attrs:
                        wildfire_src = element.get('src')
                        if 'wildfireapp' in wildfire_src:
                            wildfire_inbound = 1
                            print("Wildfire_inbound is there")

            if(omniture_inbound==1):
                pass
            else:
                # search for google ad sense pixel
                resultOmniture_inbound= element.find_all(string=re.compile('omni',re.IGNORECASE), recursive=True)
                if resultOmniture_inbound:
                    omniture_inbound = 1
                    print("Omniture_inbound is there")
                else:
                    if 'src' in element.attrs:
                        omniture_src = element.get('src')
                        if 'omniture' in omniture_src:
                            omniture_inbound = 1
                            print("Omniture_inbound is there")
            if(google_tag_manager_inbound==1):
                pass
            else:
                resultGoogleTagManager = element.find_all(string=re.compile('gtm.js',re.IGNORECASE), recursive=True)
                # print(resultGoogleTagManager)
                if resultGoogleTagManager:
                    google_tag_manager_inbound = 1
                    print("Google Tag Manager is there")
                    # print(resultGoogleTagManager[0])
                    resultGTM = re.findall(r"((GTM)(\-)[\w]{6,7})",resultGoogleTagManager[0])
                    # print(resultGTM)
                    # print(resultGTM[0])
                    if resultGTM:
                        dig_google_tag_manager_inbound = 1
                        # print(dig_google_tag_manager_inbound)
                        # print(resultGTM)
                        dig_gtm_url = "https://www.googletagmanager.com/gtm.js?id="+resultGTM[0][0]
                        print(dig_gtm_url)
            if(adobe_tag_manager_inbound==1):
                pass
            else:
                resultAdobe = element.find_all(string=re.compile('amc.on',re.IGNORECASE), recursive=True)
                resultAdobe_2 = element.find_all(string=re.compile('amc.js',re.IGNORECASE), recursive=True)
                if resultAdobe:
                    adobe_tag_manager_inbound = 1
                    print("Adobe Tag Manager is there")
                elif resultAdobe_2:
                    adobe_tag_manager_inbound = 1
                    print("Adobe Tag Manager is there")
                else:
                    pass
            if(adobe_tag_manager_outbound==1):
                pass
            else:
                resultAdobeOutbound = element.find_all(string=re.compile('adobe.com',re.IGNORECASE), recursive=True)
                if resultAdobeOutbound:
                    adobe_tag_manager_outbound = 1
                    print("Adobe Tag Manager Outbound is there")
            if(bright_tag_manager==1):
                pass
            else:
                resultBrightTag = element.find_all(string=re.compile('brighttag',re.IGNORECASE), recursive=True)
                if resultBrightTag:
                    bright_tag_manager = 1
                    print("Bright Tag Manager is there")
                else:
                    if 'src' in element.attrs:
                        bright_tag_src = element.get('src')
                        # print(criteo_src)
                        if 'brighttag' in bright_tag_src:
                            bright_tag_manager = 1
                            print("Bright Tag Manager is there")
            if(tealium_manager==1):
                pass
            else:
                resultTealium = element.find_all(string=re.compile('utag.js',re.IGNORECASE), recursive=True)
                resultTealium_2 = element.find_all(string=re.compile('tealium',re.IGNORECASE), recursive=True)
                if resultTealium:
                    tealium_manager = 1
                    print("Tealium Manager is there")
                elif resultTealium_2:
                    tealium_manager = 1
                    print("Tealium Manager is there")
                else:
                    pass
            if(tagman_manager==1):
                pass
            else:   
                resultTagman = element.find_all(string=re.compile('tagman.com',re.IGNORECASE), recursive=True)
                if resultTagman:
                    tagman_manager = 1
                    print("Tagman Manager is there")

            if(magento_flag==1):
                pass
            else:
                resultMagento = element.find_all(string=re.compile('magento',re.IGNORECASE), recursive=True)
                if resultMagento:
                    magento_flag = 1
                    print("Magento is there")
            if(paypalpay_flag==1):
                pass
            else:
                resultPaypal = element.find_all(string=re.compile('paypal',re.IGNORECASE), recursive=True)
                if resultPaypal:
                    paypalpay_flag = 1
                    print("Paypal is there")
            if(amazon_flag==1):
                pass
            else:
                resultAmazon = element.find_all(string=re.compile('aws.com',re.IGNORECASE), recursive=True)
                resultAmazon_2 = element.find_all(string=re.compile('amazon',re.IGNORECASE), recursive=True)
                if resultAmazon:
                    amazon_flag = 1
                    print("Amazon web store is there")
                elif resultAmazon_2:
                    amazon_flag = 1
                    print("Amazon web store is there")
                else:
                    pass
            if(bigcommerce_flag==1):
                pass
            else:
                # search for google remarketing pixel
                resultBigCommerce = element.find_all(string=re.compile('bigcommerce',re.IGNORECASE), recursive=True)
                if resultBigCommerce:
                    bigcommerce_flag = 1
                    print("Big Commerce is there")
            if(squarespace_flag==1):
                pass
            else:
                # search for google remarketing pixel
                resultSquarespace = element.find_all(string=re.compile('squarespace',re.IGNORECASE), recursive=True)
                if resultSquarespace:
                    squarespace_flag = 1
                    print("Squarespace ECommerce is there")
            if(shopify_flag==1):    
                pass
            else:
                # search for google remarketing pixel
                resultShopify = element.find_all(string=re.compile('shopify',re.IGNORECASE), recursive=True)
                if resultShopify:
                    shopify_flag = 1
                    print("Shopify is there")

            # Payment Links Capturing
            if(mastercardpay_flag==1):
                pass
            else:
                resultMasterCard = element.find_all(string=re.compile('mastercard',re.IGNORECASE), recursive=True)
                if resultMasterCard:
                    mastercardpay_flag = 1
                    print("Master Card is there")
            if(visapay_flag==1):
                pass
            else:
                resultVisa = element.find_all(string=re.compile('visa',re.IGNORECASE), recursive=True)
                if resultVisa:
                    visapay_flag = 1
                    print("Visa is there")
            if(amexpay_flag==1):
                pass
            else:
                resultAmex = element.find_all(string=re.compile('amex',re.IGNORECASE), recursive=True)
                resultAmex_2 = element.find_all(string=re.compile('americanexpress',re.IGNORECASE), recursive=True)
                if resultAmex:
                    amexpay_flag = 1
                    print("Amex is there")
                elif resultAmex_2:
                    amexpay_flag = 1
                    print("Amex is there")
                else:
                    pass
            if(applepay_flag==1):
                pass
            else:
                resultApplePay = element.find_all(string=re.compile('apple-pay',re.IGNORECASE), recursive=True)
                if resultApplePay:
                    applepay_flag = 1
                    print("Apple Pay is there")
            if(googlepay_flag==1):
                pass
            else:
                resultGooglePay = element.find_all(string=re.compile('google-pay',re.IGNORECASE), recursive=True)
                if resultGooglePay:
                    googlepay_flag = 1
                    print("Google Pay is there")
            if(shopifypay_flag==1):
                pass
            else:
                resultShopifyPay = element.find_all(string=re.compile('pay.shopify.com',re.IGNORECASE), recursive=True)
                resultShopifyPay_2 = element.find_all(string=re.compile('shopifypay',re.IGNORECASE), recursive=True)
                if resultShopifyPay:
                    shopifypay_flag = 1
                    print("Shopify Pay is there")
                elif resultShopifyPay_2:
                    shopifypay_flag = 1
                    print("Shopify Pay is there")
                else:
                    if 'src' in element.attrs:
                        shopifypay_src = element.get('src')
                        if 'shopify-pay' in shopifypay_src:
                            shopifypay_flag = 1
                            print("Shopify Pay is there")
            if(masterpasspay_flag==1):
                pass
            else:
                resultMasterPassPay = element.find_all(string=re.compile('masterpass.com',re.IGNORECASE), recursive=True)
                if resultMasterPassPay:
                    masterpasspay_flag = 1
                    print("Masterpass Pay is there")
                else:
                    if 'src' in element.attrs:
                        masterpasspay_src = element.get('src')
                        if 'masterpass.com' in masterpasspay_src:
                            masterpasspay_flag = 1
                            print("Masterpass Pay is there")
            if(amazonpay_flag==1):
                pass
            else:
                resultAmazonPay = element.find_all(string=re.compile('amazon-pay',re.IGNORECASE), recursive=True)
                resultAmazonPay_2 = element.find_all(string=re.compile('amazonpay',re.IGNORECASE), recursive=True)
                resultAmazonPay_3 = element.find_all(string=re.compile('payments.amazon.com',re.IGNORECASE), recursive=True)
                if resultAmazonPay:
                    amazonpay_flag = 1
                    print("Amazon Pay is there")
                elif resultAmazonPay_2:
                    amazonpay_flag = 1
                    print("Amazon Pay is there")
                elif resultAmazonPay_3:
                    amazonpay_flag = 1
                    print("Amazon Pay is there")
                else:
                    if 'src' in element.attrs:
                        amazonpay_src = element.get('src')
                        if 'payments.amazon.com' in amazonpay_src:
                            amazonpay_flag = 1
                            print("Amazon Pay is there")
            if(stripepay_flag==1):
                pass
            else:
                resultStripePay = element.find_all(string=re.compile('stripe.com',re.IGNORECASE), recursive=True)
                if resultStripePay:
                    stripepay_flag = 1
                    print("Stripe Pay is there")
                else:
                    if 'src' in element.attrs:
                        stripepay_src = element.get('src')
                        if 'stripe.com' in stripepay_src:
                            stripepay_flag = 1
                            print("Stripe Pay is there")
            if(chasepay_flag==1):
                pass
            else:
                resultChasePay = element.find_all(string=re.compile('chasepaymentech',re.IGNORECASE), recursive=True)
                if resultChasePay:
                    chasepay_flag = 1
                    print("Chase Pay is there")

            if(dig_google_tag_manager_inbound==1):
                if(gtm_hopping==1):
                    pass
                else:
                    try:
                        gtm_req = requests.get(dig_gtm_url,headers=headers,timeout=30)
                        gtm_req.raise_for_status()
                        gtm_hopping = 1
                        print("gtm_hopping",1)
                        gtm_soup = BeautifulSoup(gtm_req.content, 'lxml')
                        if(snap_pixel==1):
                            pass
                        else:
                            resultSnap = gtm_soup.find_all(string=re.compile('snaptr',re.IGNORECASE), recursive=True)
                            if resultSnap:
                                snap_pixel = 1
                                print("GTM Snapchat pixel is there")
                        if(pinterest_pixel==1):
                            pass
                        else:
                            resultPinterest = gtm_soup.find_all(string=re.compile('pintrk',re.IGNORECASE), recursive=True)
                            if resultPinterest:
                                pinterest_pixel = 1
                                print("GTM Pinterest pixel is there")
                        if(facebook_pixel==1):
                            pass
                        else:
                            resultFacebook = gtm_soup.find_all(string=re.compile('fbq', re.IGNORECASE),recursive=True)
                            if resultFacebook:
                                if 'fbq(' in resultFacebook[0]:
                                    facebook_pixel = 1
                                    print("GTM Facebook pixel is there")
                                elif '_fbq' in resultFacebook[0]:
                                    facebook_pixel = 1
                                    print("GTM Facebook pixel is there")
                                else:
                                    pass
                        if(twitter_pixel==1):
                            pass
                        else:
                            resultTwitter = gtm_soup.find_all(string=re.compile('twq', re.IGNORECASE),recursive=True)
                            if resultTwitter:
                                if 'twq(' in resultTwitter[0]:
                                    twitter_pixel = 1
                                    print("GTM Twitter pixel is there")
                                elif '_twq' in resultTwitter[0]:
                                    twitter_pixel = 1
                                    print("GTM Twitter pixel is there")
                                else:
                                    pass
                        if(criteo_pixel==1):
                            pass
                        else:
                            resultCriteo = gtm_soup.find_all(string=re.compile('criteo_',re.IGNORECASE), recursive=True)
                            if resultCriteo:
                                criteo_pixel = 1
                                print("GTM Criteo pixel is there")
                    except Exception:
                        pass

            if 'src' in element.attrs:
                ads_src = element.get('src')
                # print(adroll_src)
                if 'bat.bing.com' in ads_src:
                    bing_ad_services_flag = 1
                    bing_ad_count = bing_ad_count+1
                elif 'amazon-adsystem.com' in ads_src:
                    amazon_ad_services_flag = 1
                    amazon_ad_count = amazon_ad_count + 1
                elif 'd.adroll.com' in ads_src:
                    if 'aol' in ads_src:
                        aol_ad_services_flag = 1
                        aol_ad_count = aol_ad_count + 1
                else:
                    pass

        # for script in soup(["script", "style"]): # remove all javascript and stylesheet code
        for script in soup(["script", "style", "meta", "noscript"]): # remove all javascript and stylesheet code
            script.extract()


        # for link in soup.findAll('a', attrs={'href': re.compile(r'(^https://)')}):
        # for link in soup.findAll('a', attrs={'href': re.compile(r'(^https://)|(^http://)')}):
        for link in soup.findAll(href=True):
            # print(link)
            if(shopify_flag==1):
                pass
            else:
                if "shopify" in link.get('href'):
                    shopify_flag = 1
            if(magento_flag==1):
                pass
            else:
                if "magento" in link.get('href'):
                    magento_flag = 1
            if(paypalpay_flag==1):
                pass
            else:
                if "paypal" in link.get('href'):
                    paypalpay_flag = 1
            if(amazon_flag==1):
                pass
            else:
                # if "amazon" in link.get('href'):
                if any(x in link.get('href') for x in ["amazon","aws.com"]):
                    amazon_flag = 1
            if(bigcommerce_flag==1):
                pass
            else:
                if "bigcommerce" in link.get('href'):
                    bigcommerce_flag = 1
            if(squarespace_flag==1):
                pass
            else:
                if "squarespace" in link.get('href'):
                    squarespace_flag = 1
            if(mastercardpay_flag==1):
                pass
            else:
                if "mastercard" in link.get('href'):
                    mastercardpay_flag = 1
            if(visapay_flag==1):
                pass
            else:
                if "visa" in link.get('href'):
                    visapay_flag = 1
            if(amexpay_flag==1):
                pass
            else:
                if any(x in link.get('href') for x in ["amex","americanexpress"]):
                # if "amex" in link.get("href"):
                    amexpay_flag = 1
            if(applepay_flag==1):
                pass
            else:
                if "apple-pay" in link.get('href'):
                    applepay_flag = 1
            if(googlepay_flag==1):
                pass
            else:
                if "google-pay" in link.get('href'):
                    googlepay_flag = 1
            if(shopifypay_flag==1):
                pass
            else:
                if "pay.shopify" in link.get('href'):
                    shopifypay_flag = 1
            if(masterpasspay_flag==1):
                pass
            else:
                # if any(x in link.get('href') for x in ["amex","americanexpress"]):
                if "masterpass.com" in link.get("href"):
                    masterpasspay_flag = 1
            if(amazonpay_flag==1):
                pass
            else:
                if "payments.amazon" in link.get('href'):
                    amazonpay_flag = 1
            if(stripepay_flag==1):
                pass
            else:
                if "stripe.com" in link.get('href'):
                    stripepay_flag = 1
            if(chasepay_flag==1):
                pass
            else:
                if "chasepaymentech" in link.get('href'):
                    chasepay_flag = 1

            if(android_flag==1):
                pass
            else:
                if "play.google.com/store/apps" in link.get('href'):
                    android_flag = 1
                    android_link = link.get('href')
                # elif "android" in link.get('href'):
                #     android_flag = 1
                #     android_link = link.get('href')
                else:
                    pass
            if(itunes_flag==1):
                pass
            else:
                if "itunes.apple" in link.get('href'):
                    app_identifier_list = ['/app/','Fapp','/MZStore.woa/wa/','/developer/']
                    if any(x in link.get('href') for x in app_identifier_list):
                        itunes_flag = 1
                        itunes_url = link.get('href')
                        if('/app/' in itunes_url):
                            itunes_app_flag = 1
                            resultItunes = re.findall(r"((id)[\w]{9,10})",itunes_url)
                            if resultItunes:
                                app_id = resultItunes[0][0]
                                app_id = app_id[2:]
                            else:
                                resultItunes = re.findall(r"([&?](id=)[\w]{9,10})",itunes_url)
                                app_id = resultItunes[0][0]
                                app_id = app_id[3:]
                        if('Fapp' in itunes_url):
                            itunes_app_flag = 1
                            resultItunes = re.findall(r"((id)[\w]{9,10})",itunes_url)
                            if resultItunes:
                                app_id = resultItunes[0][0]
                                app_id = app_id[2:]
                            else:
                                resultItunes = re.findall(r"([&?](id=)[\w]{9,10})",itunes_url)
                                app_id = resultItunes[0][0]
                                app_id = app_id[3:]
                        if('/MZStore.woa/wa/' in itunes_url):
                            itunes_app_flag = 1
                            resultItunes = re.findall(r"((id)[\w]{9,10})",itunes_url)
                            if resultItunes:
                                app_id = resultItunes[0][0]
                                app_id = app_id[2:]
                            else:
                                resultItunes = re.findall(r"([&?](id=)[\w]{9,10})",itunes_url)
                                app_id = resultItunes[0][0]
                                app_id = app_id[3:]
                        if('/developer/' in itunes_url):
                            itunes_developer_flag = 1
                            # parsing out developer ID
                            resultItunes = re.findall(r"((id)[\w]{9,10})",itunes_url)
                            if resultItunes:
                                app_id = resultItunes[0][0]
                                app_id = app_id[2:]
                            else:
                                resultItunes = re.findall(r"([&?](id=)[\w]{9,10})",itunes_url)
                                app_id = resultItunes[0][0]
                                app_id = app_id[3:]

                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
                        req_itunes = requests.get(itunes_url,headers=headers,timeout=100)
                        req_itunes.raise_for_status()
                        soup_itunes =BeautifulSoup(req_itunes.content, 'lxml')
                        print(itunes_url)

                        app_title = soup_itunes.find("h1",{"class":"product-header__title app-header__title"})
                        if app_title:
                            app_title = app_title.get_text().strip().split("\n")
                            app_title = app_title[0]
                        app_subtitle = soup_itunes.find("h2",{"class":"product-header__subtitle app-header__subtitle"})
                        if app_subtitle:
                            app_subtitle = app_subtitle.get_text().strip()
                        app_identity = soup_itunes.find("h2",{"class":"product-header__identity app-header__identity"})
                        if app_identity:
                            app_identity = app_identity.get_text().strip()
                        # print(app_title,app_subtitle,app_identity)
                        app_ranking_details = soup_itunes.findAll("ul",{"class":"product-header__list app-header__list"})
                        counter = -1
                        for element in app_ranking_details:
                            counter = counter+1
                            if(counter==0):
                                app_ranking_desc = element.find("li",{"class":"product-header__list__item"})
                                if app_ranking_desc:
                                    app_ranking_desc_inline = app_ranking_desc.find("ul",{"class":"inline-list inline-list--mobile-compact"})
                                    app_ranking_soup = app_ranking_desc_inline.find("li",{"class":"inline-list__item"})
                                    if app_ranking_soup:
                                        app_ranking = app_ranking_soup.get_text().strip().split(" in ")[0]
                                        if('#' in app_ranking):
                                            pass
                                        else:
                                            app_ranking = ''
                            else:
                                app_ranking_desc = element.find("li",{"class":"product-header__list__item"})
                                if app_ranking_desc:
                                    app_ranking_desc_inline = app_ranking_desc.find("ul",{"class":"inline-list inline-list--mobile-compact"})
                                    app_price_soup = app_ranking_desc_inline.find("li",{"class":"inline-list__item inline-list__item--bulleted"})
                                    app_purchase_soup  = app_ranking_desc_inline.find("li",{"class":"inline-list__item inline-list__item--bulleted app-header__list__item--in-app-purchase"})
                                    if app_price_soup:
                                        app_price = app_price_soup.get_text().strip()
                                    if app_purchase_soup:
                                        app_purchase = app_purchase_soup.get_text().strip()
                        app_description_raw = soup_itunes.find("div",{"class":"section__description"})
                        if app_description_raw:
                            app_description = app_description_raw.find("p").get_text().strip()
                            print(app_description)
                        app_rating = soup_itunes.find("span",{"class":"we-customer-ratings__averages__display"})
                        if app_rating:
                            app_rating = app_rating.get_text().strip()
                            print(app_rating)
                        app_rating_count = soup_itunes.find("div",{"class":"we-customer-ratings__count small-hide medium-show"})
                        if app_rating_count:
                            app_rating_count = app_rating_count.get_text().strip()
                            app_rating_count = app_rating_count.split(" ")
                            app_rating_count = app_rating_count[0]
                            print(app_rating_count)
                        app_information = soup_itunes.findAll("dd",{"class":"information-list__item__definition l-column medium-9 large-6"})
                        # print(app_information)
                        app_seller = app_information[0].get_text().strip()
                        # print(app_seller)
                        app_size = app_information[1].get_text().strip()
                        # print(app_size)
                        app_category = app_information[2].get_text().strip()
                        # print(app_category)
                        app_age_rating = app_information[3].get_text().strip()
                        # print(app_age_rating)
                else:
                    pass

            if(snapchat_badge==1):
                pass
            else:
                if "snapchat.com" in link.get('href'):
                    snapchat_badge = 1
            if(pinterest_badge==1):
                pass
            else:
                if "pinterest.com" in link.get('href'):
                    pinterest_badge = 1
                elif "pinterest.co.uk" in link.get('href'):
                    pinterest_badge = 1
                elif "pinterest.fr" in link.get('href'):
                    pinterest_badge = 1
                else:
                    pass
            if(facebook_badge==1):
                pass
            else:
                if "facebook.com" in link.get('href'):
                    facebook_badge = 1
            if(instagram_badge==1):
                pass
            else:
                if "instagram.com" in link.get('href'):
                    insta_url = link.get('href')
                    instagram_badge = 1
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
                    insta_req = requests.get(insta_url,headers=headers,timeout=100)
                    insta_req.raise_for_status()
                    soup_insta =BeautifulSoup(insta_req.content, 'lxml')
                    element=soup_insta.find('meta', {'name':'description'})
                    if 'content' in element.attrs:
                        insta_handle = element.get('content')
                        # print(insta_handle)
                        insta_handle_list = insta_handle.split('@')
                        handle = insta_handle_list[-1]
                        if handle[-1]==")":
                            handle = handle[:len(handle)-1]
                        hashtag_url = "https://instagram.com/explore/tags/{}".format(handle)
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
                        print(hashtag_url)
                        hash_req = requests.get(hashtag_url,headers=headers,timeout=30)
                        hash_req.raise_for_status()
                        hash_soup =BeautifulSoup(hash_req.content, 'lxml')
                        hash_section = hash_soup.find('meta', {'name':'description'})
                        if 'content' in hash_section.attrs:
                            hash_content = hash_section.get('content')
                            hash_content_2 = hash_content.replace('-',',')
                            hash_list = hash_content_2.split(', ')
                            # print(insta_list)
                            hash_posts = hash_list[0].split(' ')
                            hash_posts = hash_posts[0]
                            hash_posts = hash_posts.replace(',', '')
                            if 'k' in hash_posts:
                                hash_posts = float(hash_posts.replace('k',''))*1000
                            elif 'm' in hash_posts:
                                hash_posts = float(hash_posts.replace('m',''))*1000000
                            elif 'b' in hash_posts:
                                hash_posts = float(hash_posts.replace('b',''))*1000000000
                            else:
                                hash_posts=float(hash_posts)
                        insta_content = element.get('content')
                        insta_content_2 = insta_content.replace('-',',')
                        insta_list = insta_content_2.split(', ')
                        # print(insta_list)
                        Insta_Followers = insta_list[0].split(' ')
                        Insta_Following = insta_list[1].split(' ')
                        Insta_Posts = insta_list[2].split(' ')
                        Insta_Followers = Insta_Followers[0]
                        Insta_Followers = Insta_Followers.replace(',', '')
                        if 'k' in Insta_Followers:
                            Insta_Followers = float(Insta_Followers.replace('k',''))*1000
                        elif 'm' in Insta_Followers:
                            Insta_Followers = float(Insta_Followers.replace('m',''))*1000000
                        elif 'b' in Insta_Followers:
                            Insta_Followers = float(Insta_Followers.replace('b',''))*1000000000
                        else:
                            Insta_Followers = float(Insta_Followers)
                        # Insta_Followers = Insta_Followers.replace('k', '000').replace('m','000000').replace('b','000000000')
                        Insta_Following = Insta_Following[0]
                        Insta_Following = Insta_Following.replace(',', '')
                        if 'k' in Insta_Following:
                            Insta_Following = float(Insta_Following.replace('k',''))*1000
                        elif 'm' in Insta_Following:
                            Insta_Following = float(Insta_Following.replace('m',''))*1000000
                        elif 'b' in Insta_Following:
                            Insta_Following = float(Insta_Following.replace('b',''))*1000000000
                        else:
                            Insta_Following = float(Insta_Following)
                        # Insta_Following = Insta_Following.replace('k', '000').replace('m','000000').replace('b','000000000')
                        Insta_Posts = Insta_Posts[0]
                        Insta_Posts = Insta_Posts.replace(',', '')
                        if 'k' in Insta_Posts:
                            Insta_Posts = float(Insta_Posts.replace('k',''))*1000
                        elif 'm' in Insta_Posts:
                            Insta_Posts = float(Insta_Posts.replace('m',''))*1000000
                        elif 'b' in Insta_Posts:
                            Insta_Posts = float(Insta_Posts.replace('b',''))*1000000000
                        else:
                            Insta_Posts = float(Insta_Posts)

            if(twitter_badge==1):
                pass
            else:
                if "twitter.com" in link.get('href'):
                    twitter_url = link.get('href')
                    twitter_badge = 1

                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
                    twitter_req = requests.get(twitter_url,headers=headers,timeout=100)
                    twitter_req.raise_for_status()
                    twitter_soup =BeautifulSoup(twitter_req.content, 'lxml')
                    soup_Followers=  twitter_soup.find('li', {'class':'ProfileNav-item--followers'})
                    if soup_Followers:
                        soup_Followers_2 = soup_Followers.find('span', {'class':'ProfileNav-value'})
                        if soup_Followers_2:
                            Followers = soup_Followers_2.text
                            Followers = Followers.strip()
                            Followers = Followers.replace(',', '')
                            if any(x in Followers for x in ['k','K']):
                            # if 'k' or 'K' in Followers:
                                Followers = float(Followers.replace('k','').replace('K',''))*1000
                            elif any(x in Followers for x in ['m','M']):
                                Followers = float(Followers.replace('m','').replace('M',''))*1000000
                            elif any(x in Followers for x in ['b','B']):
                                Followers = float(Followers.replace('b','').replace('B',''))*1000000000
                            else:
                                pass
                    soup_Following=  twitter_soup.find('li', {'class':'ProfileNav-item--following'})
                    if soup_Following:
                        soup_Following_2 = soup_Following.find('span', {'class':'ProfileNav-value'})
                        if soup_Following_2:
                            Following = soup_Following_2.text
                            Following = Following.strip()
                            Following = Following.replace(',', '')
                            if any(x in Following for x in ['k','K']):
                                Following = float(Following.replace('k','').replace('K',''))*1000
                            elif any(x in Following for x in ['m','M']):
                                Following = float(Following.replace('m','').replace('M',''))*1000000
                            elif any(x in Following for x in ['b','B']):
                                Following = float(Following.replace('b','').replace('B',''))*1000000000
                            else:
                                pass
                    soup_Likes=  twitter_soup.find('li', {'class':'ProfileNav-item--favorites'})
                    if soup_Likes:
                        soup_Likes_2 = soup_Likes.find('span', {'class':'ProfileNav-value'})
                        if soup_Likes_2:
                            Likes = soup_Likes_2.text
                            Likes = Likes.strip()
                            Likes = Likes.replace(',', '')
                            if any(x in Likes for x in ['k','K']):
                                Likes = float(Likes.replace('k','').replace('K',''))*1000
                            elif any(x in Likes for x in ['m','M']):
                                Likes = float(Likes.replace('m','').replace('M',''))*1000000
                            elif any(x in Likes for x in ['b','B']):
                                Likes = float(Likes.replace('b','').replace('B',''))*1000000000
                            else:
                                pass
                    soup_Tweets=  twitter_soup.find('li', {'class':'ProfileNav-item--tweets'})
                    if soup_Tweets:
                        soup_Tweets_2 = soup_Tweets.find('span', {'class':'ProfileNav-value'})
                        if soup_Tweets_2:
                            Tweets = soup_Tweets_2.text
                            Tweets = Tweets.strip()
                            Tweets = Tweets.replace(',', '')
                            if any(x in Tweets for x in ['k','K']):
                                Tweets = float(Tweets.replace('k','').replace('K',''))*1000
                            elif any(x in Tweets for x in ['m','M']):
                                Tweets = float(Tweets.replace('m','').replace('M',''))*1000000
                            elif any(x in Tweets for x in ['b','B']):
                                Tweets = float(Tweets.replace('b','').replace('B',''))*1000000000
                            else:
                                pass
                    print("Twitter_Followers",Followers)
                    print("Twitter_Following",Following)
                    print("Twitter_Likes",Likes)
                    print("Twitter_Tweets",Tweets)
                    comments = twitter_soup.find_all('span', attrs={'class':'ProfileTweet-actionCountForAria'})
                    if comments:
                        Post_Replies = comments[0].contents
                        # print(Post_Replies)
                        Post_Replies = Post_Replies[0].strip(' ')
                        Post_Replies = Post_Replies[0]
                        if any(x in Post_Replies for x in ['k','K']):
                            Post_Replies = float(Post_Replies.replace('k','').replace('K',''))*1000
                        elif any(x in Post_Replies for x in ['m','M']):
                            Post_Replies = float(Post_Replies.replace('m','').replace('M',''))*1000000
                        elif any(x in Post_Replies for x in ['b','B']):
                            Post_Replies = float(Post_Replies.replace('b','').replace('B',''))*1000000000
                        else:
                            pass
                        Post_Retweets = comments[1].contents
                        Post_Retweets = Post_Retweets[0].strip(' ')
                        Post_Retweets = Post_Retweets[0]
                        if any(x in Post_Retweets for x in ['k','K']):
                            Post_Retweets = float(Post_Retweets.replace('k','').replace('K',''))*1000
                        elif any(x in Post_Retweets for x in ['m','M']):
                            Post_Retweets = float(Post_Retweets.replace('m','').replace('M',''))*1000000
                        elif any(x in Post_Retweets for x in ['b','B']):
                            Post_Retweets = float(Post_Retweets.replace('b','').replace('B',''))*1000000000
                        else:
                            pass
                        Post_Likes = comments[2].contents
                        Post_Likes = Post_Likes[0].strip(' ')
                        Post_Likes = Post_Likes[0]
                        if any(x in Post_Likes for x in ['k','K']):
                            Post_Likes = float(Post_Likes.replace('k','').replace('K',''))*1000
                        elif any(x in Post_Likes for x in ['m','M']):
                            Post_Likes = float(Post_Likes.replace('m','').replace('M',''))*1000000
                        elif any(x in Post_Likes for x in ['b','B']):
                            Post_Likes = float(Post_Likes.replace('b','').replace('B',''))*1000000000
                        else:
                            pass
                    else:
                        Post_Replies,Post_Retweets,Post_Likes = '','',''

            if(linkedin_badge==1):
                pass
            else:
                if "linkedin.com" in link.get('href'):
                    linkedin_badge = 1
            if(yelp_badge==1):
                pass
            else:
                if "yelp.com" in link.get('href'):
                    yelp_badge = 1
            if(youtube_badge==1):
                pass
            else:
                if "youtube.com" in link.get('href'):
                    youtube_badge = 1
            if(google_badge==1):
                pass
            else:
                if "google.com" in link.get('href'):
                    google_badge = 1
            if "as.y.atwola" in link.get('href'):
                yahoo_ad_services_flag = 1
                yahoo_ad_count = yahoo_ad_count + 1
            elif "bing.com/aclick" in link.get('href'):
                bing_ad_services_flag = 1
                bing_ad_count = bing_ad_count+1
            else:
                pass

        # img_soup = soup.find_all('img')
        for img_element in soup.find_all('img'):
            if 'src' in img_element.attrs:
                img_src = img_element.get('src')
                # img_src = a['src']
                if('amazon-pay' in img_src):
                    amazonpay_flag = 1
                elif('amazonpay' in img_src):
                    amazonpay_flag = 1
                elif('stripe.com' in img_src):
                    stripepay_flag = 1
                elif('masterpass' in img_src):
                    masterpasspay_flag = 1
                elif('shopify' in img_src):
                    shopifypay_flag = 1
                elif('paypal' in img_src):
                    paypalpay_flag = 1
                elif('apple-pay' in img_src):
                    applepay_flag = 1
                elif('google-pay' in img_src):
                    googlepay_flag = 1
                elif('visa' in img_src):
                    visapay_flag = 1
                elif('amex' in img_src):
                    amexpay_flag = 1
                elif('mastercard' in img_src):
                    mastercardpay_flag = 1
                elif('discover' in img_src):
                    discoverypay_flag = 1
                elif('jcb' in img_src):
                    jcbpay_flag = 1
                elif('sagepay' in img_src):
                    sagepay_flag = 1
                elif("s.yimg.com" in img_src):
                    yahoo_ad_services_flag = 1
                    yahoo_ad_count = yahoo_ad_count + 1
                elif("o.aolcdn.com" in img_src):
                    aol_ad_services_flag = 1
                    aol_ad_count = aol_ad_count + 1
                else:
                    pass

        for iFrame in soup.find_all('iframe'):
            if 'src' in iFrame.attrs:
                adsense_src = iFrame.get('src')
                print(adsense_src)
                if "s.yimg.com" in adsense_src:
                    print("Found")
                    yahoo_ad_services_flag = 1
                    yahoo_ad_count = yahoo_ad_count + 1
                if "o.aolcdn.com" in adsense_src:
                    print("Found")
                    aol_ad_services_flag = 1
                    aol_ad_count = aol_ad_count + 1
                if "amazon-adsystem.com" in adsense_src:
                    amazon_ad_services_flag = 1
                    amazon_ad_count = amazon_ad_count + 1
                if 'bat.bing.com' in adsense_src:
                    bing_ad_services_flag = 1
                    bing_ad_count = bing_ad_count+1

        for ins_tag in soup.find_all('ins'):
            if 'class' in ins_tag.attrs:
                googleadsense_class = ins_tag.get('class')
                if "adsbygoogle" in googleadsense_class:
                    google_ad_services_flag = 1
                    google_ad_count = google_ad_count + 1

        searched_word_list=['form','Form','FORM']
        Leadgen_final_list = []
        for searched_word in searched_word_list:
            Leadgen_im_form = 0
            form_list=[]
            results=soup.find_all(searched_word)
            Info=''
            for t in results:
                for ti in t.find_all('input'):
                    Leadgen_im_form = 1
                    val=ti.get('placeholder')
                    if val:
                        low_val = val.lower()
                        if 'search' in low_val:
                            Leadgen_im_form = 0
                        form_list.append(Leadgen_im_form)
            if(len(form_list)>0):
                Leadgen_im_form = max(form_list)
            Leadgen_final_list.append(Leadgen_im_form)

        Leadgen_form = max(Leadgen_final_list)

        Keywords_Subarchetype_Subscription = ['subscription box']
        Keywords_Subarchetype_PetFood = ['pet', 'dog', 'cat', 'treats']
        Keywords_Subarchetype_FastFashion = ['shop collection', 'fashion', 'new arrivals', 'jeans', 'dresses', 'dress', 'shoes', 'tops']
        Keywords_Subarchetype_JewelryAndAccessories = ['jewelry', 'bracelets', 'necklaces']
        Keywords_Subarchetype_CustomizedGifts = ['customized gifts', 'unique gift', 'handmade', 'personal gift', 'custom gift', 'independent sellers']
        Keywords_Subarchetype_HomeGoods = ['home goods', 'doorbells', 'alarm system', 'handmade', 'personal gifts']
        Keywords_Subarchetype_Technology = ['laptop', 'software', 'hardware']
        Keywords_Subarchetype_Travel = ['travel', 'ticketing', 'book trip', 'booking', 'book a trip', 'vacation','hotel', 'explore', 'car rental', 'airport']
        Keywords_Subarchetype_Sustainable = ['sustainable clothing','sustainable makeup','sustainable','sustainability','cruelty-free','fair trade','non-gmo']
        Keywords_Subarchetype_Beauty = ['makeup','make-up','beauty products','hair product','skin care','cosmetics','perfume','fragrance','perfume','beauty']
        Keywords_Subarchetype_FestivalAndMusicEvents = ['music festival','summer music','festivals','live bands','concerts']
        Keywords_Subarchetype_VideoGames = ['video games','playstation','xbox','fortnite']
        Keywords_Subarchetype_MoviesAndEntertainment = ['movies','entertainment']
        Keywords_Subarchetype_SportsAndFitness = ['fitness','sports']
        Keywords_Subarchetype_ToysAndHobbies = ['plush','hobby store','toys','hobbies']

        for searched_word in Keywords_Subarchetype_Subscription:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_Subscription = 1
        for searched_word in Keywords_Subarchetype_PetFood:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_PetFood = 1
        for searched_word in Keywords_Subarchetype_FastFashion:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_FastFashion = 1
        for searched_word in Keywords_Subarchetype_JewelryAndAccessories:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_JewelryAndAccessories = 1
        for searched_word in Keywords_Subarchetype_CustomizedGifts:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_CustomizedGifts = 1
        for searched_word in Keywords_Subarchetype_HomeGoods:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_HomeGoods = 1
        for searched_word in Keywords_Subarchetype_Technology:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_Technology = 1
        for searched_word in Keywords_Subarchetype_Travel:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_Travel = 1
        for searched_word in Keywords_Subarchetype_Sustainable:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_Sustainable = 1
        for searched_word in Keywords_Subarchetype_Beauty:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_Beauty = 1
        for searched_word in Keywords_Subarchetype_FestivalAndMusicEvents:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_FestivalAndMusicEvents = 1
        for searched_word in Keywords_Subarchetype_VideoGames:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_VideoGames = 1
        for searched_word in Keywords_Subarchetype_MoviesAndEntertainment:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_MoviesAndEntertainment = 1
        for searched_word in Keywords_Subarchetype_SportsAndFitness:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_SportsAndFitness = 1
        for searched_word in Keywords_Subarchetype_ToysAndHobbies:
            results = soup.body.find_all(string=re.compile('.*{0}.*'.format(searched_word),re.IGNORECASE), recursive=True)
            if results:
                ECom_Subarch_ToysAndHobbies = 1

        for script in soup(["header","footer","p"]):
            script.extract()

        soup_text = soup.get_text()
        productLinkList = []
        productCategoryList = []
        productString = ""
        testList = ["collection","product"]
        for link in soup.findAll('a',href=True):
            # print(productLink)
            if any(x in link.get('href') for x in testList):
            # if "collections" in link.get('href'):
                linkSplitList = link.get('href').split('/')
                if linkSplitList:
                    if linkSplitList[-1]  not in productString:
                        productString = productString+linkSplitList[-1]+" "
                        productCategoryList.append(linkSplitList[-1])
                productLink = link.get('href')
                if('http' in productLink):
                    productLinkList.append(productLink)
                else:
                    productLinkList.append(url.lower()+productLink)

        # print(productLinkList)
        uniqueProductLinkList = list(set(productLinkList))
        uniqueProductCategoryList = list(set(productCategoryList))

        results = re.findall(r"((US ){0,1}(Rs\.|RS\.|\$|\€|\£|Rs|₹|INR|USD|EUR|CAD|C\$){1}(\s){0,1}[\d,]+(\.\d+){0,1}(\s){0,1}(INR|AED|USD|CAD){0,1})",soup_text)
        priceList = []
        currencySymbolList = []

        for result in results:
            every_price = result[0]
            dotList = every_price.split('.')
            if (len(dotList) > 1):
                afterDot = dotList[-1]
                if(len(afterDot)>2):
                    every_price_trimmed = every_price[:-len(afterDot)+2]
                    price = price_str(every_price_trimmed,default='0')
                else:
                    price = price_str(every_price,default='0')
            else:
                price = price_str(every_price,default='0')
            float_price = float(price)
            if float(price)==0:
                pass
            else:
                print(float_price)
                priceList.append(float_price)
                currencySym = every_price
                if any(x in currencySym for x in rupeeList):
                    currencySymbol = 'Rs.'
                    currencySymbolList.append(currencySym)
                elif any(x in currencySym for x in dollarList):
                    currencySymbol = '$'
                    currencySymbolList.append(currencySym)
                elif any(x in currencySym for x in euroList):
                    currencySymbol = '€'
                    currencySymbolList.append(currencySym)
                elif any(x in currencySym for x in poundList):
                    currencySymbol = '£'
                    currencySymbolList.append(currencySym)
                elif any(x in currencySym for x in CADList):
                    currencySymbol = 'CAD'
                    currencySymbolList.append(currencySym)
                else:
                    currencySymbol = currencySym[0]
                    currencySymbolList.append(currencySym)
        if priceList:
            max_price = max(priceList)
            min_price = min(priceList)
            avg_price = sum(priceList)/len(priceList)
            price_count = len(priceList)
        elif uniqueProductLinkList:
            list_1 = uniqueProductLinkList[0]
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
            req_2 = requests.get(list_1,headers=headers,timeout=10)
            req_2.raise_for_status()
            soup_2 = BeautifulSoup(req_2.content, 'lxml')  
            for script in soup_2(["script", "style", "meta", "noscript","header","footer","p"]):
                script.extract()
            soup2_text = soup_2.get_text()
            price_crawled_flag = 1
            # results = re.findall(r"((US ){0,1}(Rs\.|RS\.|\$|\€|\£|Rs|RS|₹|INR|USD|EUR|CAD|C\$){1}(\s){0,1}[\d,]+(\.\d+){0,1}(\s){0,1}(INR|AED|USD|CAD){0,1})",soup2_text)
            results = re.findall(r"((US ){0,1}(Rs\.|RS\.|\$|\€|\£|Rs|₹|INR|USD|EUR|CAD|C\$){1}(\s){0,1}[\d,]+(\.\d+){0,1}(\s){0,1}(INR|AED|USD|CAD){0,1})",soup2_text)
            priceList = []
            currencySymbolList = []
            for result in results:
                every_price = result[0]
                dotList = every_price.split('.')
                if (len(dotList) > 1):
                    afterDot = dotList[-1]
                    if(len(afterDot)>2):
                        every_price_trimmed = every_price[:-len(afterDot)+2]
                        price = price_str(every_price_trimmed,default='0')
                    else:
                        price = price_str(every_price,default='0')
                else:
                    price = price_str(every_price,default='0')
                float_price = float(price)
                if float(price)==0:
                    pass
                else:
                    print(float_price)
                    priceList.append(float_price)
                    currencySym = every_price
                    if any(x in currencySym for x in rupeeList):
                        currencySymbol = 'Rs.'
                        currencySymbolList.append(currencySym)
                    elif any(x in currencySym for x in dollarList):
                        currencySymbol = '$'
                        currencySymbolList.append(currencySym)
                    elif any(x in currencySym for x in euroList):
                        currencySymbol = '€'
                        currencySymbolList.append(currencySym)
                    elif any(x in currencySym for x in poundList):
                        currencySymbol = '£'
                        currencySymbolList.append(currencySym)
                    elif any(x in currencySym for x in CADList):
                        currencySymbol = 'CAD'
                        currencySymbolList.append(currencySym)
                    else:
                        currencySymbol = currencySym[0]
                        currencySymbolList.append(currencySym)
            if priceList:
                max_price = max(priceList)
                min_price = min(priceList)
                avg_price = sum(priceList)/len(priceList)
                price_count = len(priceList)
        else:
            pass

        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([crawled_flag_website],columns=['crawled_flag_website']),pd.DataFrame([shopify_flag],columns=['shopify_flag']),
            pd.DataFrame([android_flag],columns=['android_flag']),pd.DataFrame([android_link],columns=['android_link']),pd.DataFrame([itunes_flag],columns=['itunes_flag']),
            pd.DataFrame([cart_hopping],columns=['cart_hopping']),pd.DataFrame([checkout_hopping],columns=['checkout_hopping']),pd.DataFrame([Leadgen_form],columns=['Leadgen_form']),
            pd.DataFrame([ECom_Subarch_Subscription],columns=["ECom_Subarch_Subscription"]),pd.DataFrame([ECom_Subarch_PetFood],columns=["ECom_Subarch_PetFood"]),pd.DataFrame([ECom_Subarch_FastFashion],columns=["ECom_Subarch_FastFashion"]),pd.DataFrame([ECom_Subarch_JewelryAndAccessories],columns=["ECom_Subarch_JewelryAndAccessories"]),pd.DataFrame([ECom_Subarch_CustomizedGifts],columns=["ECom_Subarch_CustomizedGifts"]),pd.DataFrame([ECom_Subarch_HomeGoods],columns=["ECom_Subarch_HomeGoods"]),pd.DataFrame([ECom_Subarch_Technology],columns=["ECom_Subarch_Technology"]),pd.DataFrame([ECom_Subarch_Travel],columns=["ECom_Subarch_Travel"]),pd.DataFrame([ECom_Subarch_Sustainable],columns=["ECom_Subarch_Sustainable"]),pd.DataFrame([ECom_Subarch_Beauty],columns=["ECom_Subarch_Beauty"]),pd.DataFrame([ECom_Subarch_FestivalAndMusicEvents],columns=["ECom_Subarch_FestivalAndMusicEvents"]),pd.DataFrame([ECom_Subarch_VideoGames],columns=["ECom_Subarch_VideoGames"]),pd.DataFrame([ECom_Subarch_MoviesAndEntertainment],columns=["ECom_Subarch_MoviesAndEntertainment"]),pd.DataFrame([ECom_Subarch_SportsAndFitness],columns=["ECom_Subarch_SportsAndFitness"]),pd.DataFrame([ECom_Subarch_ToysAndHobbies],columns=["ECom_Subarch_ToysAndHobbies"]),
            pd.DataFrame([magento_flag],columns=['magento_flag']),pd.DataFrame([paypalpay_flag],columns=['paypalpay_flag']),pd.DataFrame([amazon_flag],columns=['amazon_flag']),pd.DataFrame([bigcommerce_flag],columns=['bigcommerce_flag']),pd.DataFrame([squarespace_flag],columns=['squarespace_flag']),
            pd.DataFrame([mastercardpay_flag],columns=['mastercardpay_flag']),pd.DataFrame([visapay_flag],columns=['visapay_flag']),pd.DataFrame([amexpay_flag],columns=['amexpay_flag']),pd.DataFrame([applepay_flag],columns=['applepay_flag']),pd.DataFrame([googlepay_flag],columns=['googlepay_flag']),
            pd.DataFrame([shopifypay_flag],columns=['shopifypay_flag']),pd.DataFrame([masterpasspay_flag],columns=['masterpasspay_flag']),pd.DataFrame([amazonpay_flag],columns=['amazonpay_flag']),pd.DataFrame([stripepay_flag],columns=['stripepay_flag']),
            pd.DataFrame([chasepay_flag],columns=['chasepay_flag']),pd.DataFrame([discoverypay_flag],columns=['discoverypay_flag']),pd.DataFrame([jcbpay_flag],columns=['jcbpay_flag']),pd.DataFrame([sagepay_flag],columns=['sagepay_flag']),
            pd.DataFrame([snap_pixel],columns=['snap_pixel']),pd.DataFrame([pinterest_pixel],columns=['pinterest_pixel']),pd.DataFrame([facebook_pixel],columns=['facebook_pixel']),pd.DataFrame([twitter_pixel],columns=['twitter_pixel']),pd.DataFrame([criteo_pixel],columns=['criteo_pixel']),pd.DataFrame([google_ad_sense_inbound],columns=['google_ad_sense_inbound']),pd.DataFrame([google_remarketing_inbound],columns=['google_remarketing_inbound']),
            pd.DataFrame([fb_exchange_inbound],columns=['fb_exchange_inbound']),pd.DataFrame([ad_roll_inbound],columns=['ad_roll_inbound']),pd.DataFrame([perfect_audience_inbound],columns=['perfect_audience_inbound']),pd.DataFrame([wildfire_inbound],columns=['wildfire_inbound']),pd.DataFrame([omniture_inbound],columns=['omniture_inbound']), 
            pd.DataFrame([google_tag_manager_inbound],columns=['google_tag_manager_inbound']),pd.DataFrame([adobe_tag_manager_inbound],columns=['adobe_tag_manager_inbound']),
            pd.DataFrame([google_ad_sense_outbound],columns=['google_ad_sense_outbound']),pd.DataFrame([google_remarketing_outbound],columns=['google_remarketing_outbound']),
            pd.DataFrame([fb_exchange_outbound],columns=['fb_exchange_outbound']),pd.DataFrame([ad_roll_outbound],columns=['ad_roll_outbound']),pd.DataFrame([perfect_audience_outbound],columns=['perfect_audience_outbound']),pd.DataFrame([wildfire_outbound],columns=['wildfire_outbound']),pd.DataFrame([omniture_outbound],columns=['omniture_outbound']),
            pd.DataFrame([google_tag_manager_outbound],columns=['google_tag_manager_outbound']),pd.DataFrame([adobe_tag_manager_outbound],columns=['adobe_tag_manager_outbound']),pd.DataFrame([bright_tag_manager],columns=['bright_tag_manager']),pd.DataFrame([tealium_manager],columns=['tealium_manager']),pd.DataFrame([tagman_manager],columns=['tagman_manager']),
            pd.DataFrame([snapchat_badge],columns=['snapchat_badge']),pd.DataFrame([pinterest_badge],columns=['pinterest_badge']),pd.DataFrame([facebook_badge],columns=['facebook_badge']),pd.DataFrame([instagram_badge],columns=['instagram_badge']),pd.DataFrame([twitter_badge],columns=['twitter_badge']),pd.DataFrame([linkedin_badge],columns=['linkedin_badge']),pd.DataFrame([yelp_badge],columns=['yelp_badge']),pd.DataFrame([youtube_badge],columns=['youtube_badge']),pd.DataFrame([google_badge],columns=['google_badge']),
            pd.DataFrame([google_ad_services_flag],columns=['google_ad_services_flag']),pd.DataFrame([google_ad_count],columns=['google_ad_count']),pd.DataFrame([yahoo_ad_services_flag],columns=['yahoo_ad_services_flag']),pd.DataFrame([yahoo_ad_count],columns=['yahoo_ad_count']),pd.DataFrame([aol_ad_services_flag],columns=['aol_ad_services_flag']),
            pd.DataFrame([aol_ad_count],columns=['aol_ad_count']),pd.DataFrame([bing_ad_services_flag],columns=['bing_ad_services_flag']),pd.DataFrame([bing_ad_count],columns=['bing_ad_count']),pd.DataFrame([amazon_ad_services_flag],columns=['amazon_ad_services_flag']),pd.DataFrame([amazon_ad_count],columns=['amazon_ad_count']),
            pd.DataFrame([price_count],columns=['price_count']),pd.DataFrame([productString],columns=['product_categories']),pd.DataFrame([currencySymbol],columns=['currency_symbol']),pd.DataFrame([min_price],columns=['min_price']),pd.DataFrame([max_price],columns=['max_price']),pd.DataFrame([avg_price],columns=['avg_price']),
            pd.DataFrame([Followers],columns=['Twitter_Followers']),pd.DataFrame([Following],columns=['Twitter_Following']),pd.DataFrame([Likes],columns=['Twitter_Likes']),pd.DataFrame([Tweets],columns=['Twitter_Tweets']),pd.DataFrame([Post_Replies],columns=['Twitter_Post_Replies']),pd.DataFrame([Post_Retweets],columns=['Twitter_Post_Retweets']),pd.DataFrame([Post_Likes],columns=['Twitter_Post_Likes']),
            pd.DataFrame([Insta_Followers],columns=['Insta_Followers']),pd.DataFrame([Insta_Following],columns=['Insta_Following']),pd.DataFrame([Insta_Posts],columns=['Insta_Posts']),pd.DataFrame([hash_posts],columns=['Insta_Hashtag_Posts']),
            pd.DataFrame([itunes_url],columns=['itunes_app_link']),pd.DataFrame([itunes_app_flag],columns=['itunes_app_flag']),pd.DataFrame([itunes_developer_flag],columns=['itunes_developer_flag']),pd.DataFrame([app_id],columns=['itunes_app_id']),pd.DataFrame([app_title],columns=['itunes_app_title']),pd.DataFrame([app_subtitle],columns=['itunes_app_subtitle']),pd.DataFrame([app_identity],columns=['itunes_app_identity']),pd.DataFrame([app_ranking],columns=['itunes_app_ranking']),pd.DataFrame([app_price],columns=['itunes_app_price']),
            pd.DataFrame([app_purchase],columns=['itunes_app_purchase']),pd.DataFrame([app_description],columns=['itunes_app_description']),pd.DataFrame([app_rating],columns=['itunes_app_rating']),pd.DataFrame([app_rating_count],columns=['itunes_app_rating_count']),pd.DataFrame([app_seller],columns=['itunes_app_seller']),pd.DataFrame([app_size],columns=['itunes_app_size']),pd.DataFrame([app_category],columns=['itunes_app_category']),pd.DataFrame([app_age_rating],columns=['itunes_app_age_rating'])],axis=1)
        df=df.append(temp)

    except Exception:
        temp=pd.concat([pd.DataFrame([url],columns=['Website']),pd.DataFrame([crawled_flag_website],columns=['crawled_flag_website']),pd.DataFrame([shopify_flag],columns=['shopify_flag']),
            pd.DataFrame([android_flag],columns=['android_flag']),pd.DataFrame([android_link],columns=['android_link']),pd.DataFrame([itunes_flag],columns=['itunes_flag']),
            pd.DataFrame([cart_hopping],columns=['cart_hopping']),pd.DataFrame([checkout_hopping],columns=['checkout_hopping']),pd.DataFrame([Leadgen_form],columns=['Leadgen_form']),
            pd.DataFrame([ECom_Subarch_Subscription],columns=["ECom_Subarch_Subscription"]),pd.DataFrame([ECom_Subarch_PetFood],columns=["ECom_Subarch_PetFood"]),pd.DataFrame([ECom_Subarch_FastFashion],columns=["ECom_Subarch_FastFashion"]),pd.DataFrame([ECom_Subarch_JewelryAndAccessories],columns=["ECom_Subarch_JewelryAndAccessories"]),pd.DataFrame([ECom_Subarch_CustomizedGifts],columns=["ECom_Subarch_CustomizedGifts"]),pd.DataFrame([ECom_Subarch_HomeGoods],columns=["ECom_Subarch_HomeGoods"]),pd.DataFrame([ECom_Subarch_Technology],columns=["ECom_Subarch_Technology"]),pd.DataFrame([ECom_Subarch_Travel],columns=["ECom_Subarch_Travel"]),pd.DataFrame([ECom_Subarch_Sustainable],columns=["ECom_Subarch_Sustainable"]),pd.DataFrame([ECom_Subarch_Beauty],columns=["ECom_Subarch_Beauty"]),pd.DataFrame([ECom_Subarch_FestivalAndMusicEvents],columns=["ECom_Subarch_FestivalAndMusicEvents"]),pd.DataFrame([ECom_Subarch_VideoGames],columns=["ECom_Subarch_VideoGames"]),pd.DataFrame([ECom_Subarch_MoviesAndEntertainment],columns=["ECom_Subarch_MoviesAndEntertainment"]),pd.DataFrame([ECom_Subarch_SportsAndFitness],columns=["ECom_Subarch_SportsAndFitness"]),pd.DataFrame([ECom_Subarch_ToysAndHobbies],columns=["ECom_Subarch_ToysAndHobbies"]),
            pd.DataFrame([magento_flag],columns=['magento_flag']),pd.DataFrame([paypalpay_flag],columns=['paypalpay_flag']),pd.DataFrame([amazon_flag],columns=['amazon_flag']),pd.DataFrame([bigcommerce_flag],columns=['bigcommerce_flag']),pd.DataFrame([squarespace_flag],columns=['squarespace_flag']),
            pd.DataFrame([mastercardpay_flag],columns=['mastercardpay_flag']),pd.DataFrame([visapay_flag],columns=['visapay_flag']),pd.DataFrame([amexpay_flag],columns=['amexpay_flag']),pd.DataFrame([applepay_flag],columns=['applepay_flag']),pd.DataFrame([googlepay_flag],columns=['googlepay_flag']),
            pd.DataFrame([shopifypay_flag],columns=['shopifypay_flag']),pd.DataFrame([masterpasspay_flag],columns=['masterpasspay_flag']),pd.DataFrame([amazonpay_flag],columns=['amazonpay_flag']),pd.DataFrame([stripepay_flag],columns=['stripepay_flag']),
            pd.DataFrame([chasepay_flag],columns=['chasepay_flag']),pd.DataFrame([discoverypay_flag],columns=['discoverypay_flag']),pd.DataFrame([jcbpay_flag],columns=['jcbpay_flag']),pd.DataFrame([sagepay_flag],columns=['sagepay_flag']),
            pd.DataFrame([snap_pixel],columns=['snap_pixel']),pd.DataFrame([pinterest_pixel],columns=['pinterest_pixel']),pd.DataFrame([facebook_pixel],columns=['facebook_pixel']),pd.DataFrame([twitter_pixel],columns=['twitter_pixel']),pd.DataFrame([criteo_pixel],columns=['criteo_pixel']),pd.DataFrame([google_ad_sense_inbound],columns=['google_ad_sense_inbound']),pd.DataFrame([google_remarketing_inbound],columns=['google_remarketing_inbound']),
            pd.DataFrame([fb_exchange_inbound],columns=['fb_exchange_inbound']),pd.DataFrame([ad_roll_inbound],columns=['ad_roll_inbound']),pd.DataFrame([perfect_audience_inbound],columns=['perfect_audience_inbound']),pd.DataFrame([wildfire_inbound],columns=['wildfire_inbound']),pd.DataFrame([omniture_inbound],columns=['omniture_inbound']), 
            pd.DataFrame([google_tag_manager_inbound],columns=['google_tag_manager_inbound']),pd.DataFrame([adobe_tag_manager_inbound],columns=['adobe_tag_manager_inbound']),
            pd.DataFrame([google_ad_sense_outbound],columns=['google_ad_sense_outbound']),pd.DataFrame([google_remarketing_outbound],columns=['google_remarketing_outbound']),
            pd.DataFrame([fb_exchange_outbound],columns=['fb_exchange_outbound']),pd.DataFrame([ad_roll_outbound],columns=['ad_roll_outbound']),pd.DataFrame([perfect_audience_outbound],columns=['perfect_audience_outbound']),pd.DataFrame([wildfire_outbound],columns=['wildfire_outbound']),pd.DataFrame([omniture_outbound],columns=['omniture_outbound']),
            pd.DataFrame([google_tag_manager_outbound],columns=['google_tag_manager_outbound']),pd.DataFrame([adobe_tag_manager_outbound],columns=['adobe_tag_manager_outbound']),pd.DataFrame([bright_tag_manager],columns=['bright_tag_manager']),pd.DataFrame([tealium_manager],columns=['tealium_manager']),pd.DataFrame([tagman_manager],columns=['tagman_manager']),
            pd.DataFrame([snapchat_badge],columns=['snapchat_badge']),pd.DataFrame([pinterest_badge],columns=['pinterest_badge']),pd.DataFrame([facebook_badge],columns=['facebook_badge']),pd.DataFrame([instagram_badge],columns=['instagram_badge']),pd.DataFrame([twitter_badge],columns=['twitter_badge']),pd.DataFrame([linkedin_badge],columns=['linkedin_badge']),pd.DataFrame([yelp_badge],columns=['yelp_badge']),pd.DataFrame([youtube_badge],columns=['youtube_badge']),pd.DataFrame([google_badge],columns=['google_badge']),
            pd.DataFrame([google_ad_services_flag],columns=['google_ad_services_flag']),pd.DataFrame([google_ad_count],columns=['google_ad_count']),pd.DataFrame([yahoo_ad_services_flag],columns=['yahoo_ad_services_flag']),pd.DataFrame([yahoo_ad_count],columns=['yahoo_ad_count']),pd.DataFrame([aol_ad_services_flag],columns=['aol_ad_services_flag']),
            pd.DataFrame([aol_ad_count],columns=['aol_ad_count']),pd.DataFrame([bing_ad_services_flag],columns=['bing_ad_services_flag']),pd.DataFrame([bing_ad_count],columns=['bing_ad_count']),pd.DataFrame([amazon_ad_services_flag],columns=['amazon_ad_services_flag']),pd.DataFrame([amazon_ad_count],columns=['amazon_ad_count']),
            pd.DataFrame([price_count],columns=['price_count']),pd.DataFrame([productString],columns=['product_categories']),pd.DataFrame([currencySymbol],columns=['currency_symbol']),pd.DataFrame([min_price],columns=['min_price']),pd.DataFrame([max_price],columns=['max_price']),pd.DataFrame([avg_price],columns=['avg_price']),
            pd.DataFrame([Followers],columns=['Twitter_Followers']),pd.DataFrame([Following],columns=['Twitter_Following']),pd.DataFrame([Likes],columns=['Twitter_Likes']),pd.DataFrame([Tweets],columns=['Twitter_Tweets']),pd.DataFrame([Post_Replies],columns=['Twitter_Post_Replies']),pd.DataFrame([Post_Retweets],columns=['Twitter_Post_Retweets']),pd.DataFrame([Post_Likes],columns=['Twitter_Post_Likes']),
            pd.DataFrame([Insta_Followers],columns=['Insta_Followers']),pd.DataFrame([Insta_Following],columns=['Insta_Following']),pd.DataFrame([Insta_Posts],columns=['Insta_Posts']),pd.DataFrame([hash_posts],columns=['Insta_Hashtag_Posts']),
            pd.DataFrame([itunes_url],columns=['itunes_app_link']),pd.DataFrame([itunes_app_flag],columns=['itunes_app_flag']),pd.DataFrame([itunes_developer_flag],columns=['itunes_developer_flag']),pd.DataFrame([app_id],columns=['itunes_app_id']),pd.DataFrame([app_title],columns=['itunes_app_title']),pd.DataFrame([app_subtitle],columns=['itunes_app_subtitle']),pd.DataFrame([app_identity],columns=['itunes_app_identity']),pd.DataFrame([app_ranking],columns=['itunes_app_ranking']),pd.DataFrame([app_price],columns=['itunes_app_price']),
            pd.DataFrame([app_purchase],columns=['itunes_app_purchase']),pd.DataFrame([app_description],columns=['itunes_app_description']),pd.DataFrame([app_rating],columns=['itunes_app_rating']),pd.DataFrame([app_rating_count],columns=['itunes_app_rating_count']),pd.DataFrame([app_seller],columns=['itunes_app_seller']),pd.DataFrame([app_size],columns=['itunes_app_size']),pd.DataFrame([app_category],columns=['itunes_app_category']),pd.DataFrame([app_age_rating],columns=['itunes_app_age_rating'])],axis=1)
        df=df.append(temp)

    return(str(df.values.tolist()))

# callback table creation
@app.callback(Output('datatable-1','children'),
              [Input('upload-data-file', 'contents'),
              Input('upload-data-file', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        # dff = pd.read_csv(io.StringIO(content))
        # return dff.to_dict('rows')
        df = parse_contents(contents, filename)
        
        if df is not None:
            return html.Div([
                dt.DataTable(data=df.to_dict('rows'),columns=[{'id': c, 'name': c} for c in df.columns],

                    style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])
                # html.Hr(),
                # html.Div('Raw Content'),
                # html.Pre(contents, style=pre_style)
            ])
        else:
            return html.Div([
                dt.DataTable(data=[{}])])

# callback table creation
@app.callback(Output('datatable-L1','children'),
              [Input('upload-data-file-L1', 'contents'),
              Input('upload-data-file-L1', 'filename')])
def update_output_L(contents, filename):
    if contents is not None:
        # dff = pd.read_csv(io.StringIO(content))
        # return dff.to_dict('rows')
        df = parse_contents(contents, filename)
        
        if df is not None:
            return html.Div([
                dt.DataTable(data=df.to_dict('rows'),columns=[{'id': c, 'name': c} for c in df.columns],

                    style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])
                # html.Hr(),
                # html.Div('Raw Content'),
                # html.Pre(contents, style=pre_style)
            ])
        else:
            return html.Div([
                dt.DataTable(data=[{}])])


def channel_assignment(webCrawledData):

    # To calculate category score
    category_conditions = [
    webCrawledData['Ecom Subarch Beauty'] == 1,
    webCrawledData['Ecom Subarch Customized Gifts'] == 1,
    webCrawledData['Ecom Subarch Fast Fashion'] == 1,
    webCrawledData['Ecom Subarch Jewelry And Accessories'] == 1,
    webCrawledData['Ecom Subarch Festival And Music Events'] == 1,
    webCrawledData['Ecom Subarch Movies And Entertainment'] == 1,
    webCrawledData['Ecom Subarch Sports And Fitness'] == 1,
    webCrawledData['Ecom Subarch Subscription'] == 1,
    webCrawledData['Ecom Subarch Sustainable'] == 1,
    webCrawledData['Ecom Subarch Toys And Hobbies'] == 1,
    webCrawledData['Ecom Subarch Pet Food'] == 1,
    webCrawledData['Ecom Subarch Technology'] == 1,
    webCrawledData['Ecom Subarch Video Games'] == 1,
    webCrawledData['Ecom Subarch Travel'] == 1,
    webCrawledData['Ecom Subarch Home Goods'] == 1,
    ]

    category_outputs = [3.1,3.5,3.9,4.2,3.6,2.6,2,5,3,2.3,2.8,3.8,3.6,2.8,4.3]

    webCrawledData['Category Score'] = np.select(category_conditions,category_outputs,0)

    # To calculate PRICE POINT SCORE
    webCrawledData["Avg Price"] = webCrawledData["Avg Price"].fillna(0, inplace=True)
    webCrawledData['Avg Price'] = webCrawledData['Avg Price'].str.extract('(\d+)')
    webCrawledData["Avg Price"].fillna(0, inplace=True)
    webCrawledData['Avg Price'].astype(int)
    
    price_point_conditions = [
    webCrawledData['Avg Price'].gt(0)&webCrawledData['Avg Price'].le(10),
    webCrawledData['Avg Price'].gt(10)&webCrawledData['Avg Price'].le(25),
    webCrawledData['Avg Price'].gt(25)&webCrawledData['Avg Price'].le(50),
    webCrawledData['Avg Price'].gt(50)&webCrawledData['Avg Price'].le(100),
    webCrawledData['Avg Price'].gt(100)&webCrawledData['Avg Price'].le(1000),
    webCrawledData['Avg Price'].gt(1000),
    ]

    price_point_outputs = [2.7,4.2,4.4,5,3.5,-1]

    webCrawledData['Price Point Score'] = np.select(price_point_conditions,price_point_outputs,0)

    # To calculate Millenial Badge SCORE

    millenial_badge_conditions = [
    webCrawledData[['Pinterest Badge','Instagram Badge','Snapchat Badge']].sum(1).gt(0),
    ]

    millenial_badge_outputs = [5]

    webCrawledData['Badge Millenium Score'] = np.select(millenial_badge_conditions,millenial_badge_outputs,0)

    # To calculate Pixel Score

    pixel_conditions = [
    webCrawledData[['Snap Pixel','Facebook Pixel','Pinterest Pixel','Criteo Pixel','Twitter Pixel']].sum(1).eq(0),
    ]
    pixel_outputs = [1]
    webCrawledData['No Pixel Flag'] = np.select(pixel_conditions,pixel_outputs,0)

    webCrawledData['Pixel Score'] = webCrawledData.apply(pixelScore,axis=1) 

    # To calculate ecommerce_stack score

    ecommerce_stack_conditions = [
    webCrawledData[["Amazon Flag","Bigcommerce Flag","Magento Flag","Shopify Flag","Squarespace Flag"]].sum(1).gt(0),
    ]

    ecommerce_stack_outputs = [5]

    webCrawledData['Ecomm Stack Score'] = np.select(ecommerce_stack_conditions,ecommerce_stack_outputs,0)

    # To calculate payment score

    webCrawledData['Ecomm Payment Score'] = webCrawledData.apply(paymentScore,axis=1)

    # To calculate ad tech score

    webCrawledData['Ad Tech Score'] = webCrawledData.apply(adTechScore,axis=1)

    # To calculate yelp score

    webCrawledData['Yelp Score'] = webCrawledData.apply(yelpScore,axis=1)

    # webCrawledData["Twitter_Followers"] = webCrawledData["Twitter_Followers"].fillna(0, inplace=True)
    # webCrawledData["Twitter_Likes"] = webCrawledData["Twitter_Likes"].fillna(0, inplace=True)
    # webCrawledData["Twitter_Tweets"] = webCrawledData["Twitter_Tweets"].fillna(0, inplace=True)
    # webCrawledData["Twitter_Following"] = webCrawledData["Twitter_Following"].fillna(0, inplace=True)
    # webCrawledData["Twitter_Post_Replies"] = webCrawledData["Twitter_Post_Replies"].fillna(0, inplace=True)
    # webCrawledData["Twitter_Post_Retweets"] = webCrawledData["Twitter_Post_Retweets"].fillna(0, inplace=True)
    # webCrawledData["Twitter_Post_Likes"] = webCrawledData["Twitter_Post_Likes"].fillna(0, inplace=True)
    print(webCrawledData["Twitter Followers"])
    webCrawledData['Twitter Followers'] = webCrawledData['Twitter Followers'].astype(str)
    webCrawledData['Twitter Following'] = webCrawledData['Twitter Following'].astype(str)
    webCrawledData['Twitter Likes'] = webCrawledData['Twitter Likes'].astype(str)
    webCrawledData['Twitter Tweets'] = webCrawledData['Twitter Tweets'].astype(str)
    webCrawledData['Twitter Post Replies'] = webCrawledData['Twitter Post Replies'].astype(str)
    webCrawledData['Twitter Post Retweets'] = webCrawledData['Twitter Post Retweets'].astype(str)
    webCrawledData['Twitter Post Likes'] = webCrawledData['Twitter Post Likes'].astype(str)
    print(webCrawledData["Twitter Followers"])

    # print(webCrawledData["Twitter Followers"].dtypes)
    webCrawledData['Twitter Followers'] = webCrawledData['Twitter Followers'].str.extract('(\d+)')
    webCrawledData['Twitter Following'] = webCrawledData['Twitter Following'].str.extract('(\d+)')
    webCrawledData['Twitter Likes'] = webCrawledData['Twitter Likes'].str.extract('(\d+)')
    webCrawledData['Twitter Tweets'] = webCrawledData['Twitter Tweets'].str.extract('(\d+)')
    webCrawledData['Twitter Post Replies'] = webCrawledData['Twitter Post Replies'].str.extract('(\d+)')
    webCrawledData['Twitter Post Retweets'] = webCrawledData['Twitter Post Retweets'].str.extract('(\d+)')
    webCrawledData['Twitter Post Likes'] = webCrawledData['Twitter Post Likes'].str.extract('(\d+)')
    print(webCrawledData["Twitter Followers"])

    webCrawledData["Insta Followers"] = webCrawledData["Insta Followers"].astype(str)
    webCrawledData["Insta Posts"] = webCrawledData["Insta Posts"].astype(str)
    webCrawledData["Insta Hashtag Posts"] = webCrawledData["Insta Hashtag Posts"].astype(str)
    webCrawledData["Insta Following"] = webCrawledData["Insta Following"].astype(str)
    webCrawledData["Insta Followers"] = webCrawledData["Insta Followers"].str.extract('(\d+)')
    webCrawledData["Insta Posts"] = webCrawledData["Insta Posts"].str.extract('(\d+)')
    webCrawledData["Insta Hashtag Posts"] = webCrawledData["Insta Hashtag Posts"].str.extract('(\d+)')
    webCrawledData["Insta Following"] = webCrawledData["Insta Following"].str.extract('(\d+)')
    # webCrawledData['Twitter Followers'] = np.where(np.isnan(webCrawledData['Twitter Followers']), 0, webCrawledData['Twitter Followers'])
    webCrawledData = webCrawledData.fillna(0)

    print(webCrawledData["Twitter Followers"])

    # np.isnan(np.array([np.nan, 0], dtype=np.float64))

    # To fill nan as zero for instagram and twitter variables
    # webCrawledData["Twitter Followers"] = webCrawledData["Twitter Followers"].fillna(0, inplace=True)
    # webCrawledData["Twitter Likes"] = webCrawledData["Twitter Likes"].fillna(0, inplace=True)
    # webCrawledData["Twitter Tweets"] = webCrawledData["Twitter Tweets"].fillna(0, inplace=True)
    # webCrawledData["Twitter Following"] = webCrawledData["Twitter Following"].fillna(0, inplace=True)
    # webCrawledData["Twitter Post Replies"] = webCrawledData["Twitter Post Replies"].fillna(0, inplace=True)
    # webCrawledData["Twitter Post Retweets"] = webCrawledData["Twitter Post Retweets"].fillna(0, inplace=True)
    # webCrawledData["Twitter Post Likes"] = webCrawledData["Twitter Post Likes"].fillna(0, inplace=True)

    # print(webCrawledData["Twitter Followers"])

    # webCrawledData = np.where(np.isnan(webCrawledData), 0, webCrawledData)
    # To calculate twitter and instagram page
    # print(webCrawledData["Twitter Followers"])

    webCrawledData["Twitter Followers"] = webCrawledData["Twitter Followers"].astype(int)
    webCrawledData["Twitter Likes"] = webCrawledData["Twitter Likes"].astype(int)
    webCrawledData["Twitter Tweets"] = webCrawledData["Twitter Tweets"].astype(int)
    webCrawledData["Twitter Following"] = webCrawledData["Twitter Following"].astype(int)
    webCrawledData["Twitter Post Replies"] = webCrawledData["Twitter Post Replies"].astype(int)
    webCrawledData["Twitter Post Retweets"] = webCrawledData["Twitter Post Retweets"].astype(int)
    webCrawledData["Twitter Post Likes"] = webCrawledData["Twitter Post Likes"].astype(int)

    print(webCrawledData["Twitter Followers"])


    # webCrawledData["Insta Followers"] = webCrawledData["Insta Followers"].fillna(0, inplace=True)
    # webCrawledData["Insta Following"] = webCrawledData["Insta Following"].fillna(0, inplace=True)
    # webCrawledData["Insta Posts"] = webCrawledData["Insta Posts"].fillna(0, inplace=True)
    # webCrawledData["Insta Hashtag Posts"] = webCrawledData["Insta Hashtag Posts"].fillna(0, inplace=True)
    
    # webCrawledData['Insta Followers'] = webCrawledData['Insta Followers'].str.extract('(\d+)')
    # webCrawledData['Insta Following'] = webCrawledData['Insta Following'].str.extract('(\d+)')
    # webCrawledData['Insta Posts'] = webCrawledData['Insta Posts'].str.extract('(\d+)')
    # webCrawledData['Insta Hashtag Posts'] = webCrawledData['Insta Hashtag Posts'].str.extract('(\d+)')

    # # To fill nan as zero for instagram and twitter variables
    # webCrawledData["Insta Followers"].fillna(0, inplace=True)
    # webCrawledData["Insta Posts"].fillna(0, inplace=True)
    # webCrawledData["Insta Hashtag Posts"].fillna(0, inplace=True)
    # webCrawledData["Insta Following"].fillna(0, inplace=True)

    webCrawledData['Insta Followers'] = webCrawledData['Insta Followers'].astype(int)
    webCrawledData['Insta Following'] = webCrawledData['Insta Following'].astype(int)
    webCrawledData['Insta Posts'] = webCrawledData['Insta Posts'].astype(int)
    webCrawledData['Insta Hashtag Posts'] = webCrawledData['Insta Hashtag Posts'].astype(int)

    twitter_page_conditions = [
    webCrawledData[["Twitter Followers","Twitter Likes",'Twitter Tweets',"Twitter Post Replies","Twitter Post Retweets"]].sum(1).gt(0),
    ]
    twitter_page_outputs = [1]
    webCrawledData['Twitter Page'] = np.select(twitter_page_conditions,twitter_page_outputs,0)
    insta_page_conditions = [
    webCrawledData[["Insta Followers","Insta Posts","Insta Hashtag Posts"]].sum(1).gt(0),
    ]
    insta_page_outputs = [1]
    webCrawledData['Insta Page'] = np.select(insta_page_conditions,insta_page_outputs,0)

    # To calculate twitter and instagram influencer
    twitter_influencer_conditions = [
    webCrawledData['Twitter Followers'].gt(0)&webCrawledData['Twitter Followers'].le(1000),
    webCrawledData['Twitter Followers'].gt(1000)&webCrawledData['Twitter Followers'].le(10000),
    webCrawledData['Twitter Followers'].gt(10000)&webCrawledData['Twitter Followers'].le(1000000),
    webCrawledData['Twitter Followers'].gt(1000000),
    ]
    twitter_influencer_outputs = [1,2,3,4]
    webCrawledData['Influencer Twitter'] = np.select(twitter_influencer_conditions,twitter_influencer_outputs,0)
    insta_influencer_conditions = [
    webCrawledData['Insta Followers'].gt(0)&webCrawledData['Insta Followers'].le(1000),
    webCrawledData['Insta Followers'].gt(1000)&webCrawledData['Insta Followers'].le(10000),
    webCrawledData['Insta Followers'].gt(10000)&webCrawledData['Insta Followers'].le(1000000),
    webCrawledData['Insta Followers'].gt(1000000),
    ]
    insta_influencer_outputs = [1,2,3,4]
    webCrawledData['Influencer Insta'] = np.select(insta_influencer_conditions,insta_influencer_outputs,0)

    # To calculate twitter and instagram engagement rate
    webCrawledData['Twitter Engagement Rate'] = webCrawledData.apply(twitterER,axis=1)
    webCrawledData['Insta Engagement Rate'] = webCrawledData.apply(instaER,axis=1)

    # To calculate twitter and instagram engagement score
    twitter_engagement_conditions = [
    webCrawledData['Twitter Engagement Rate'].gt(0)&webCrawledData['Twitter Engagement Rate'].le(0.02),
    webCrawledData['Twitter Engagement Rate'].gt(0.02)&webCrawledData['Twitter Engagement Rate'].le(0.09),
    webCrawledData['Twitter Engagement Rate'].gt(0.09)&webCrawledData['Twitter Engagement Rate'].le(0.33),
    webCrawledData['Twitter Engagement Rate'].gt(0.33),
    ]
    twitter_engagement_outputs = [1,2,3,4]
    webCrawledData['Engagement Twitter'] = np.select(twitter_engagement_conditions,twitter_engagement_outputs,0)
    insta_engagement_conditions = [
    webCrawledData['Insta Engagement Rate'].gt(0)&webCrawledData['Insta Engagement Rate'].le(1),
    webCrawledData['Insta Engagement Rate'].gt(1)&webCrawledData['Insta Engagement Rate'].le(3.5),
    webCrawledData['Insta Engagement Rate'].gt(3.5)&webCrawledData['Insta Engagement Rate'].le(6),
    webCrawledData['Insta Engagement Rate'].gt(6),
    ]
    insta_engagement_outputs = [1,2,3,4]
    webCrawledData['Engagement Insta'] = np.select(insta_engagement_conditions,insta_engagement_outputs,0)

    # To calculate twitter and instagram engagement score
    webCrawledData['Twitter Engagement Score'] = webCrawledData['Influencer Twitter']*webCrawledData['Engagement Twitter']
    webCrawledData['Insta Engagement Score'] = webCrawledData['Influencer Insta']*webCrawledData['Engagement Insta']

    # To calculate twitter and instagram combined influence score
    webCrawledData['Influencer Score'] = (webCrawledData['Insta Engagement Score']+webCrawledData['Twitter Engagement Score'])/(webCrawledData['Insta Page']+webCrawledData['Twitter Page'])
    webCrawledData['Influencer Score'].fillna(0, inplace=True)

    # To calculate Customer Fit and Digital Intensity Score
    webCrawledData['Customer Fit Score'] = webCrawledData['Category Score']+webCrawledData['Price Point Score']+webCrawledData['Badge Millenium Score']+webCrawledData['Influencer Score']*0.55+webCrawledData['Yelp Score']
    webCrawledData['Digital Intensity Score'] = webCrawledData['Ad Tech Score']*0.588+webCrawledData['Ecomm Stack Score']*0.5+webCrawledData['Ecomm Payment Score']*0.833+webCrawledData['Pixel Score']*0.41667

    webCrawledData = webCrawledData.sort_values(['Customer Fit Score'], ascending=False)
    num_rows = webCrawledData.shape[0]

    webCrawledData['CUSTFIT_IDENTIFIER'] = range(num_rows)
    custFit_conditions = [
    webCrawledData['CUSTFIT_IDENTIFIER'].ge(0)&webCrawledData['CUSTFIT_IDENTIFIER'].le(0.5*num_rows-1),
    webCrawledData['CUSTFIT_IDENTIFIER'].gt(0.5*num_rows-1)&webCrawledData['CUSTFIT_IDENTIFIER'].le(0.75*num_rows-1),
    webCrawledData['CUSTFIT_IDENTIFIER'].gt(0.75*num_rows-1)&webCrawledData['CUSTFIT_IDENTIFIER'].le(num_rows-1),
    ]
    custFit_outputs = ['H','M','L']
    webCrawledData['Customer Fit Assign'] = np.select(custFit_conditions,custFit_outputs,'')
    # webCrawledData.ix[webCrawledData["R Public Client Domain"]==1, ['Customer Fit Assign']] = 'L'

    webCrawledData = webCrawledData.sort_values(['Digital Intensity Score'], ascending=False)

    webCrawledData['DIGINTENSITY_IDENTIFIER'] = range(num_rows)
    digIntensity_conditions = [
    webCrawledData['DIGINTENSITY_IDENTIFIER'].ge(0)&webCrawledData['DIGINTENSITY_IDENTIFIER'].le(0.33*num_rows-1),
    webCrawledData['DIGINTENSITY_IDENTIFIER'].gt(0.33*num_rows-1)&webCrawledData['DIGINTENSITY_IDENTIFIER'].le(0.66*num_rows-1),
    webCrawledData['DIGINTENSITY_IDENTIFIER'].gt(0.66*num_rows-1)&webCrawledData['DIGINTENSITY_IDENTIFIER'].le(num_rows-1),
    ]
    digIntensity_outputs = ['H','M','L']
    webCrawledData['Digital Intensity Assign'] = np.select(digIntensity_conditions,digIntensity_outputs,'')
    # webCrawledData.ix[webCrawledData["R Public Client Domain"]==1, ['Digital Intensity Assign']] = 'L'

    # Drop the identifier column for both
    webCrawledData = webCrawledData.drop(['CUSTFIT_IDENTIFIER', 'DIGINTENSITY_IDENTIFIER'], axis=1)
    webCrawledData = webCrawledData.sort_index(axis=0)
    # To assign channel hard conditions based on DSO, Fortune 1000 and Inc 5000
    Channel_conditions = [
    # webCrawledData['R Dso Managed New'].eq(1),
    # webCrawledData['R Dso Managed New'].eq(0)&webCrawledData['Fortune 1000'].eq(1),
    # webCrawledData['R Dso Managed New'].eq(0)&webCrawledData['Fortune 1000'].eq(0)&webCrawledData['Inc 5000'].eq(1),
    webCrawledData['Customer Fit Assign'].eq('H')&webCrawledData['Digital Intensity Assign'].eq('H'),
    webCrawledData['Customer Fit Assign'].eq('H')&webCrawledData['Digital Intensity Assign'].eq('M'),
    webCrawledData['Customer Fit Assign'].eq('H')&webCrawledData['Digital Intensity Assign'].eq('L'),
    webCrawledData['Customer Fit Assign'].eq('M')&webCrawledData['Digital Intensity Assign'].eq('H'),
    webCrawledData['Customer Fit Assign'].eq('M')&webCrawledData['Digital Intensity Assign'].eq('M'),
    webCrawledData['Customer Fit Assign'].eq('M')&webCrawledData['Digital Intensity Assign'].eq('L'),
    webCrawledData['Customer Fit Assign'].eq('L')&webCrawledData['Digital Intensity Assign'].eq('H'),
    webCrawledData['Customer Fit Assign'].eq('L')&webCrawledData['Digital Intensity Assign'].eq('M'),
    webCrawledData['Customer Fit Assign'].eq('L')&webCrawledData['Digital Intensity Assign'].eq('L'),
    ]
    Channel_outputs = ['GOS','GOS','VENDOR','GOS','VENDOR','UNMANAGED','UNMANAGED','UNMANAGED','UNMANAGED']
    webCrawledData['Ecomm Channel Assigned'] = np.select(Channel_conditions,Channel_outputs,'')

    # file_name = "Web_Crawler_consolidated_03062019_3.1_"+".csv"
    # webCrawledData.to_csv(file_name, sep=',', index = False, encoding='utf-8')
    # print ("Done! Please see the csv file")
    # if webCrawledData.empty:
    #     return html.Div([dt.DataTable(data=[{}])])
    # else:
    #     webCrawledData.columns = ["Website","Text","Cleaned Text","Non English Flag"]
    # return html.Div([dt.DataTable(data=webCrawledData.to_dict('rows'))])

    # category_4_var = ["Website","Category Score","Price Point Score","Badge Millenium Score","No Pixel Flag","Pixel Score","Ecomm Stack Score","Ecomm Payment Score","Ad Tech Score","Yelp Score","Twitter Page","Insta Page","Influencer Twitter","Influencer Insta","Twitter Engagement Rate","Insta Engagement Rate","Engagement Twitter","Engagement Insta","Twitter Engagement Score","Insta Engagement Score","Influencer Score","Customer Fit Score","Digital Intensity Score","Customer Fit Assign","Digital Intensity Assign","Ecomm Channel Assigned"]
    
    return webCrawledData

def image_type(website):

    # Initializing dataframe
    df=pd.DataFrame()

    crawled_flag_website = 0
    
    try:     

        # Selenium Latest Web Driver crawling
        # chrome_options = Options()
        # chrome_options.add_argument('--dns-prefetch-disable')
        # chrome_options.add_argument('--no-sandbox')
        # # chrome_options.add_argument('--lang=en-US')
        # chrome_options.add_argument('--headless')
        # chrome_options.add_argument("--disable-logging")
        # chrome_options.add_argument('log-level=3')
        # # chrome_options.add_argument('--disable-gpu')
        # # chrome_options.add_experimental_option('prefs', {'intl.accept_languages': 'en-US'})
        # browser = webdriver.Chrome("C:\\Users\\a.daluka\\Documents\\Web Crawler\\chromedriver.exe", chrome_options=chrome_options)
        # browser.get(website)
        # soup = BeautifulSoup(browser.page_source, 'lxml')
        # browser.quit()

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}
        req = requests.get(website,headers=headers,timeout=300)
        req.raise_for_status()
        soup = BeautifulSoup(req.content, 'lxml')

        # print(soup)

        crawled_flag_website = 1

        img_tags = soup.find_all('img')

        urls = []
        # print(img_tags)
        for img in img_tags:
            if 'src' in img.attrs:
                img_src = img.get('src')
                # urls = [img['src'] for img in img_tags]
                urls.append(img_src)
        # print(urls)
        # load the VGG16 network pre-trained on the ImageNet dataset
        print("[INFO] loading network...")

        model = VGG16(weights="imagenet")

        count = 0
        check = 0
        if len(urls)>0:
            for url in urls:

                filename = re.search(r'([\w_-]+[.](jpg|png|jpeg))', url)

                if filename is None:
                    count = count+1
                else:
                    try:
                        if any(x in filename.group(1) for x in ['icon','img1x1']):
                            check = check+1
                            pass
                        else:
                            with open(filename.group(1), 'wb') as f:
                                # if 'http' not in url:
                                #         # sometimes an image source can be relative 
                                #         # if it is provide the base url which also happens 
                                #         # to be the site variable atm. 
                                #     url = '{}{}'.format(site, url)
                                response = requests.get(url,headers=headers,timeout=30)
                                f.write(response.content)
                            print("[INFO] loading and preprocessing image...")
                            print(filename.group(1))
                            image = image_utils.load_img(filename.group(1), target_size=(224, 224))
                            # print(image)
                            image = image_utils.img_to_array(image)
                            image = np.expand_dims(image, axis=0)
                            image = preprocess_input(image)

                            # classify the image
                            print("[INFO] classifying image...")
                            preds = model.predict(image)
                            P = decode_predictions(preds)

                            (imagenetID, label, prob) = P[0][0]
                            # (imagenetID, label, prob) = P[0][1]

                            print(imagenetID, label, prob)

                            if any(x in label for x in ['cleaver','web_site','hook','matchstick']):
                                check = check+1
                                pass
                            else:
                                if(prob<0.20):
                                    check = check+1
                                    pass
                                else:
                                    temp=pd.concat([pd.DataFrame([website],columns=['Website']),pd.DataFrame([filename.group(1)],columns=['Image Name']),pd.DataFrame([label],columns=['Object Label']),pd.DataFrame([str(prob*100)+"%"],columns=['Object Probabiliy'])],axis=1)
                                    df=df.append(temp)
                    except Exception:
                        check = check+1
                        pass

                if count == len(urls):
                    temp=pd.concat([pd.DataFrame([website],columns=['Website']),pd.DataFrame([""],columns=['Image Name']),pd.DataFrame([""],columns=['Object Label']),pd.DataFrame([""],columns=['Object Probabiliy'])],axis=1)
                    df=df.append(temp)

                if check == len(urls):
                    temp=pd.concat([pd.DataFrame([website],columns=['Website']),pd.DataFrame([""],columns=['Image Name']),pd.DataFrame([""],columns=['Object Label']),pd.DataFrame([""],columns=['Object Probabiliy'])],axis=1)
                    df=df.append(temp)

                # else:
                #     temp=pd.concat([pd.DataFrame([website],columns=['Website']),pd.DataFrame([""],columns=['Image Name']),pd.DataFrame([""],columns=['Object Label']),pd.DataFrame([""],columns=['Object Probabiliy'])],axis=1)
                #     df=df.append(temp)
        else:
            temp=pd.concat([pd.DataFrame([website],columns=['Website']),pd.DataFrame([""],columns=['Image Name']),pd.DataFrame([""],columns=['Object Label']),pd.DataFrame([""],columns=['Object Probabiliy'])],axis=1)
            df=df.append(temp)

    except Exception:
        temp=pd.concat([pd.DataFrame([website],columns=['Website']),pd.DataFrame([""],columns=['Image Name']),pd.DataFrame([""],columns=['Object Label']),pd.DataFrame([""],columns=['Object Probabiliy'])],axis=1)
        df=df.append(temp)

    return(str(df.values.tolist()))

@app.callback([Output('datatable-2','children'),
                Output('datatable-3','children'),
                Output('datatable-4','children'),
                Output('datatable-5','children'),
                Output('download-link-1', 'children'),
                Output('table-header-1','children'),
                Output('table-header-2','children'),
                Output('table-header-3','children'),
                Output('table-header-4','children'),
                ],
                [Input('dropdown', 'value'),
                Input('submit-button-1','n_clicks'),
                Input('submit-button-2','n_clicks')],
                [State('upload-data-file', 'contents'),
                State('upload-data-file', 'filename'),
                State('input_button_1', 'value')])
                # [Event('web_elements', 'click')])
def web_crawl(value,n_clicks_1,n_clicks_2,contents,filename,website):
    if n_clicks_2 is not None:
        if contents is not None:
            base = parse_contents(contents, filename)
            print(value)
        # if(n_clicks is not None):
        # if contents is not None:
        # print(rows)
        # base = pd.DataFrame(props['rows'])
        # print(base)
            url_list=[]
            # print(base)
            for url in base['Website']:
                url_list.append(url)
            total_websites = len(url_list)
            counter = 100
            divisor = total_websites//counter
            left_websites = total_websites%counter 
            starting=0
            result=pd.DataFrame()
            col_list = ["Website","Crawled Flag Website","Shopify Flag",
                    "Android Flag","Android Link","Itunes Flag",
                    "Cart Hopping","Checkout Hopping","Leadgen Form",
                    "Ecom Subarch Subscription","Ecom Subarch Pet Food","Ecom Subarch Fast Fashion","Ecom Subarch Jewelry And Accessories","Ecom Subarch Customized Gifts","Ecom Subarch Home Goods","Ecom Subarch Technology","Ecom Subarch Travel","Ecom Subarch Sustainable","Ecom Subarch Beauty","Ecom Subarch Festival And Music Events","Ecom Subarch Video Games","Ecom Subarch Movies And Entertainment","Ecom Subarch Sports And Fitness","Ecom Subarch Toys And Hobbies",
                    "Magento Flag","Paypal Pay Flag","Amazon Flag","Bigcommerce Flag","Squarespace Flag",
                    "Mastercard Pay Flag","Visa Pay Flag","Amex Pay Flag","Apple Pay Flag","Google Pay Flag",
                    "Shopify Pay Flag","Masterpass Pay Flag","Amazon Pay Flag","Stripe Pay Flag",
                    "Chase Pay Flag","Discovery Pay Flag","Jcb Pay Flag","Sage Pay Flag",
                    "Snap Pixel","Pinterest Pixel","Facebook Pixel","Twitter Pixel","Criteo Pixel","Google Ad Sense Inbound","Google Remarketing Inbound",
                    "Fb Exchange Inbound","Ad Roll Inbound","Perfect Audience Inbound","Wildfire Inbound","Omniture Inbound",
                    "Google Tag Manager Inbound","Adobe Tag Manager Inbound","Google Ad Sense Outbound","Google Remarketing Outbound","Fb Exchange Outbound",
                    "Ad Roll Outbound","Perfect Audience Outbound","Wildfire Outbound","Omniture Outbound","Google Tag Manager Outbound",
                    "Adobe Tag Manager Outbound","Bright Tag Manager","Tealium Manager","Tagman Manager",
                    "Snapchat Badge","Pinterest Badge","Facebook Badge","Instagram Badge","Twitter Badge","Linkedin Badge","Yelp Badge","Youtube Badge","Google Badge",
                    "Google Ad Services Flag","Google Ad Count","Yahoo Ad Services Flag","Yahoo Ad Count","Aol Ad Services Flag","Aol Ad Count","Bing Ad Services Flag","Bing Ad Count","Amazon Ad Services Flag","Amazon Ad Count",
                    "Price Count","Product Categories","Currency Symbol","Min Price","Max Price","Avg Price",
                    "Twitter Followers","Twitter Following","Twitter Likes",'Twitter Tweets',"Twitter Post Replies","Twitter Post Retweets","Twitter Post Likes",
                    "Insta Followers","Insta Following","Insta Posts","Insta Hashtag Posts",
                    "Itunes App Link",'Itunes App Flag','Itunes developer Flag','Itunes App Id','Itunes App Title','Itunes App Subtitle','Itunes App Identity','Itunes App Ranking','Itunes App Price','Itunes App Purchase','Itunes App Description','Itunes App Rating','Itunes App Rating Count','Itunes App Seller','Itunes App Size','Itunes App Category','Itunes App Age Rating']
            for i in range(divisor):
                end=starting +counter
                list_100=url_list[starting:end]
                p =Pool(processes=10)
                rt = p.map(Content_type, list_100)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = col_list
                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = col_list
                time.sleep(10)
                starting=end

                # file_name = "Web_Crawler_consolidated_03292019_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")

            end=starting + left_websites
            list_remaining=url_list[starting:end]
            p =Pool(processes=10)
            rt = p.map(Content_type, list_remaining)
            p.terminate()
            p.join()
            lol = [ast.literal_eval(r) for r in rt]
            df = [pd.DataFrame(s) for s in lol]
            res = pd.concat(df)
            if res.empty:
                pass
            else:
                res.columns = col_list

            result = result.append(res)
            if result.empty:
                pass
            else:
                result.columns = col_list

            time.sleep(10)
            starting=end

            result_fin_web_data = pd.merge(base,result,on='Website',how='left')

            # file_name = "Web_Crawler_consolidated_03292019_2.1_"+str(end)+".csv"
            # result_fin.to_csv(file_name, sep=',', index = False, encoding='utf-8')
            # print ("Done! Please see the csv file")

            # ['pixels', 'tag_managers', 'ad_tech', 'ad_count', 'cart_checkout', 'price_points', 'Ecommerce-stack', 'payments', 'social_media_presence', 'android_itunes', 'leadgen-form']
            pixel_var = ["Snap Pixel","Pinterest Pixel","Facebook Pixel","Twitter Pixel","Criteo Pixel"]
            tag_manager_var = ["Google Tag Manager Inbound","Adobe Tag Manager Inbound","Bright Tag Manager","Tealium Manager","Tagman Manager",]
            ad_tech_var = ["Google Ad Sense Inbound","Google Remarketing Inbound","Fb Exchange Inbound","Ad Roll Inbound","Perfect Audience Inbound","Wildfire Inbound","Omniture Inbound","Google Ad Sense Outbound","Google Remarketing Outbound","Fb Exchange Outbound","Ad Roll Outbound","Perfect Audience Outbound","Wildfire Outbound","Omniture Outbound","Google Tag Manager Outbound","Adobe Tag Manager Outbound"]
            ad_count_var = ["Google Ad Services Flag","Google Ad Count","Yahoo Ad Services Flag","Yahoo Ad Count","Aol Ad Services Flag","Aol Ad Count","Bing Ad Services Flag","Bing Ad Count","Amazon Ad Services Flag","Amazon Ad Count"]
            cart_checkout_var = ["Cart Hopping","Checkout Hopping"]
            price_point_var = ["Price Count","Product Categories","Currency Symbol","Min Price","Max Price","Avg Price",]
            eCommerce_stack_var =["Shopify Flag","Magento Flag","Amazon Flag","Bigcommerce Flag","Squarespace Flag",]
            payment_var = ["Mastercard Pay Flag","Visa Pay Flag","Paypal Pay Flag","Amex Pay Flag","Apple Pay Flag","Google Pay Flag","Shopify Pay Flag","Masterpass Pay Flag","Amazon Pay Flag","Stripe Pay Flag","Chase Pay Flag","Discovery Pay Flag","Jcb Pay Flag","Sage Pay Flag"]
            social_media_presence_var = ["Snapchat Badge","Pinterest Badge","Facebook Badge","Instagram Badge","Twitter Badge","Linkedin Badge","Yelp Badge","Youtube Badge","Google Badge","Twitter Followers","Twitter Following","Twitter Likes",'Twitter Tweets',"Twitter Post Replies","Twitter Post Retweets","Twitter Post Likes","Insta Followers","Insta Following","Insta Posts","Insta Hashtag Posts"]
            android_itunes_var = ["Android Flag","Android Link","Itunes Flag","Itunes App Link",'Itunes App Flag','Itunes developer Flag','Itunes App Id','Itunes App Title','Itunes App Subtitle','Itunes App Identity','Itunes App Ranking','Itunes App Price','Itunes App Purchase','Itunes App Description','Itunes App Rating','Itunes App Rating Count','Itunes App Seller','Itunes App Size','Itunes App Category','Itunes App Age Rating']
            leadgen_form_var = ["Leadgen Form"]
            eCommerce_keyword_var = ["Ecom Subarch Subscription","Ecom Subarch Pet Food","Ecom Subarch Fast Fashion","Ecom Subarch Jewelry And Accessories","Ecom Subarch Customized Gifts","Ecom Subarch Home Goods","Ecom Subarch Technology","Ecom Subarch Travel","Ecom Subarch Sustainable","Ecom Subarch Beauty","Ecom Subarch Festival And Music Events","Ecom Subarch Video Games","Ecom Subarch Movies And Entertainment","Ecom Subarch Sports And Fitness","Ecom Subarch Toys And Hobbies"]
            
            category_1 = ['pixels','tag_managers','ad_tech','ad_count']
            category_2 = ['cart_checkout','price_points','ecommerce-stack','payments','android_itunes','leadgen-form']
            category_3 = ['social_media_presence']

            # if('social_media_presence' in value):
            #     data_table_2 = result_fin[['Website',"snapchat_badge","pinterest_badge","facebook_badge","instagram_badge","twitter_badge","linkedin_badge","yelp_badge","youtube_badge","google_badge","Twitter_Followers","Twitter_Following","Twitter_Likes",'Twitter_Tweets',"Twitter_Post_Replies","Twitter_Post_Retweets","Twitter_Post_Likes","Insta_Followers","Insta_Following","Insta_Posts","Insta_Hashtag_Posts"]]
            # else:
            #     data_table_2 = pd.DataFrame([[]])

            elem_dict = {'pixels':pixel_var,
                         'tag_managers':tag_manager_var,
                         'ad_tech':ad_tech_var,
                         'ad_count':ad_count_var,
                         'cart_checkout':cart_checkout_var+eCommerce_keyword_var,
                         'price_points':price_point_var,
                         'ecommerce-stack':eCommerce_stack_var,
                         'payments':payment_var,
                         'social_media_presence':social_media_presence_var,
                         'android_itunes':android_itunes_var,
                         'leadgen-form':leadgen_form_var}

            category_1_var = ["Website"]
            for cate in category_1:
                if cate in value:
                    category_1_var = category_1_var+elem_dict[cate]
            category_2_var = ["Website"]
            for cate in category_2:
                if cate in value:
                    category_2_var = category_2_var+elem_dict[cate]
            category_3_var = ["Website"]
            for cate in category_3:
                if cate in value:
                    category_3_var = category_3_var+elem_dict[cate]
            
            print(category_1_var,category_2_var,category_3_var)

            category_4_var = ["Website","Category Score","Price Point Score","Badge Millenium Score","No Pixel Flag","Pixel Score","Ecomm Stack Score","Ecomm Payment Score","Ad Tech Score","Yelp Score","Twitter Page","Insta Page","Influencer Twitter","Influencer Insta","Twitter Engagement Rate","Insta Engagement Rate","Engagement Twitter","Engagement Insta","Twitter Engagement Score","Insta Engagement Score","Influencer Score","Customer Fit Score","Digital Intensity Score","Customer Fit Assign","Digital Intensity Assign","Ecomm Channel Assigned"]

            print(result_fin_web_data[category_3_var])
            
            final_table = channel_assignment(result_fin_web_data)

            # print(result_fin_web_data[category_3_var])

            # final_table = pd.merge(result_fin_web_data,final_table,on='Website',how='left')

            # print(final_table[category_3_var])

            csv_string = final_table.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
            
            if final_table.empty:
                return html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],

                    style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                                 style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-2'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],
                style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                             style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-3'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],
                style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                             style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-4'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],
                style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                             style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-5'),html.Div([html.A('Download Results',download="webCrawledData.csv",id='download-link-1',
                href=csv_string),]),html.Div(html.P("Pixels/Ad-Techs/Tag-Managers/Other",id='table-header-1', className='title',
                style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(
                html.P("ECommerce/App/Lead-Gen-Forms",id='table-header-2',className='title',style={'fontSize':15,'color':'black', 
                    'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Social-Media-Presence",
                    id='table-header-3',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}),
                    className="six columns"),html.Div(html.P("Channel-Assignment",id='table-header-4', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns")
            else:
                return html.Div([dt.DataTable(data=final_table[category_1_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_1_var].columns],

                    style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                                 style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-2'),html.Div([dt.DataTable(data=final_table[category_2_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_2_var].columns],
                style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                             style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-3'),html.Div([dt.DataTable(data=final_table[category_3_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_3_var].columns],
                style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                             style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-4'),html.Div([dt.DataTable(data=final_table[category_4_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_4_var].columns],
                style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                             style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-5'),html.Div([html.A('Download Results',download="webCrawledData.csv",id='download-link-1',
                href=csv_string),]),html.Div(html.P("Pixels/Ad-Techs/Tag-Managers/Other",id='table-header-1', className='title',
                style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(
                html.P("ECommerce/App/Lead-Gen-Forms",id='table-header-2',className='title',style={'fontSize':15,'color':'black', 
                    'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Social-Media-Presence",
                    id='table-header-3',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}),
                    className="six columns"),html.Div(html.P("Channel-Assignment",id='table-header-4', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns")
            
        else:
            if n_clicks_1 is not None:
                url_list = [website]
                total_websites = len(url_list)
                counter = 100
                divisor = total_websites//counter
                left_websites = total_websites%counter 
                starting=0
                result=pd.DataFrame()
                col_list = ["Website","Crawled Flag Website","Shopify Flag",
                        "Android Flag","Android Link","Itunes Flag",
                        "Cart Hopping","Checkout Hopping","Leadgen Form",
                        "Ecom Subarch Subscription","Ecom Subarch Pet Food","Ecom Subarch Fast Fashion","Ecom Subarch Jewelry And Accessories","Ecom Subarch Customized Gifts","Ecom Subarch Home Goods","Ecom Subarch Technology","Ecom Subarch Travel","Ecom Subarch Sustainable","Ecom Subarch Beauty","Ecom Subarch Festival And Music Events","Ecom Subarch Video Games","Ecom Subarch Movies And Entertainment","Ecom Subarch Sports And Fitness","Ecom Subarch Toys And Hobbies",
                        "Magento Flag","Paypal Pay Flag","Amazon Flag","Bigcommerce Flag","Squarespace Flag",
                        "Mastercard Pay Flag","Visa Pay Flag","Amex Pay Flag","Apple Pay Flag","Google Pay Flag",
                        "Shopify Pay Flag","Masterpass Pay Flag","Amazon Pay Flag","Stripe Pay Flag",
                        "Chase Pay Flag","Discovery Pay Flag","Jcb Pay Flag","Sage Pay Flag",
                        "Snap Pixel","Pinterest Pixel","Facebook Pixel","Twitter Pixel","Criteo Pixel","Google Ad Sense Inbound","Google Remarketing Inbound",
                        "Fb Exchange Inbound","Ad Roll Inbound","Perfect Audience Inbound","Wildfire Inbound","Omniture Inbound",
                        "Google Tag Manager Inbound","Adobe Tag Manager Inbound","Google Ad Sense Outbound","Google Remarketing Outbound","Fb Exchange Outbound",
                        "Ad Roll Outbound","Perfect Audience Outbound","Wildfire Outbound","Omniture Outbound","Google Tag Manager Outbound",
                        "Adobe Tag Manager Outbound","Bright Tag Manager","Tealium Manager","Tagman Manager",
                        "Snapchat Badge","Pinterest Badge","Facebook Badge","Instagram Badge","Twitter Badge","Linkedin Badge","Yelp Badge","Youtube Badge","Google Badge",
                        "Google Ad Services Flag","Google Ad Count","Yahoo Ad Services Flag","Yahoo Ad Count","Aol Ad Services Flag","Aol Ad Count","Bing Ad Services Flag","Bing Ad Count","Amazon Ad Services Flag","Amazon Ad Count",
                        "Price Count","Product Categories","Currency Symbol","Min Price","Max Price","Avg Price",
                        "Twitter Followers","Twitter Following","Twitter Likes",'Twitter Tweets',"Twitter Post Replies","Twitter Post Retweets","Twitter Post Likes",
                        "Insta Followers","Insta Following","Insta Posts","Insta Hashtag Posts",
                        "Itunes App Link",'Itunes App Flag','Itunes developer Flag','Itunes App Id','Itunes App Title','Itunes App Subtitle','Itunes App Identity','Itunes App Ranking','Itunes App Price','Itunes App Purchase','Itunes App Description','Itunes App Rating','Itunes App Rating Count','Itunes App Seller','Itunes App Size','Itunes App Category','Itunes App Age Rating']
                for i in range(divisor):
                    end=starting +counter
                    list_100=url_list[starting:end]
                    p =Pool(processes=1)
                    rt = p.map(Content_type, list_100)
                    p.terminate()
                    p.join()
                    lol = [ast.literal_eval(r) for r in rt]
                    df = [pd.DataFrame(s) for s in lol]
                    res = pd.concat(df)
                    if res.empty:
                        pass
                    else:
                        res.columns = col_list
                    result = result.append(res)
                    if result.empty:
                        pass
                    else:
                        result.columns = col_list
                    time.sleep(10)
                    starting=end

                    # file_name = "Web_Crawler_consolidated_03292019_"+str(end)+".csv"
                    # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                    # print ("Done! Please see the csv file")

                end=starting + left_websites
                list_remaining=url_list[starting:end]
                p =Pool(processes=1)
                rt = p.map(Content_type, list_remaining)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = col_list

                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = col_list

                time.sleep(10)
                starting=end

                result_fin = result
                # base = pd.DataFrame([['s_test',website,0,0,0,0]],columns = ['Sih1id','Website','Fortune 1000','Inc 5000','R Dso Managed New','R Public Client Domain'])
                # result_fin = pd.merge(base,result,on='Website',how='left')

                # file_name = "Web_Crawler_consolidated_03292019_2.1_"+str(end)+".csv"
                # result_fin.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")

                # ['pixels', 'tag_managers', 'ad_tech', 'ad_count', 'cart_checkout', 'price_points', 'ecommerce-stack', 'payments', 'social_media_presence', 'android_itunes', 'leadgen-form']
                pixel_var = ["Snap Pixel","Pinterest Pixel","Facebook Pixel","Twitter Pixel","Criteo Pixel"]
                tag_manager_var = ["Google Tag Manager Inbound","Adobe Tag Manager Inbound","Bright Tag Manager","Tealium Manager","Tagman Manager",]
                ad_tech_var = ["Google Ad Sense Inbound","Google Remarketing Inbound",
                        "Fb Exchange Inbound","Ad Roll Inbound","Perfect Audience Inbound","Wildfire Inbound","Omniture Inbound",
                        "Google Ad Sense Outbound","Google Remarketing Outbound","Fb Exchange Outbound",
                        "Ad Roll Outbound","Perfect Audience Outbound","Wildfire Outbound","Omniture Outbound","Google Tag Manager Outbound",
                        "Adobe Tag Manager Outbound"]
                ad_count_var = ["Google Ad Services Flag","Google Ad Count","Yahoo Ad Services Flag","Yahoo Ad Count","Aol Ad Services Flag","Aol Ad Count","Bing Ad Services Flag","Bing Ad Count","Amazon Ad Services Flag","Amazon Ad Count"]
                cart_checkout_var = ["Cart Hopping","Checkout Hopping"]
                price_point_var = ["Price Count","Product Categories","Currency Symbol","Min Price","Max Price","Avg Price",]
                eCommerce_stack_var =["Shopify Flag","Magento Flag","Amazon Flag","Bigcommerce Flag","Squarespace Flag",]
                payment_var = ["Mastercard Pay Flag","Visa Pay Flag","Paypal Pay Flag","Amex Pay Flag","Apple Pay Flag","Google Pay Flag","Shopify Pay Flag","Masterpass Pay Flag","Amazon Pay Flag","Stripe Pay Flag","Chase Pay Flag","Discovery Pay Flag","Jcb Pay Flag","Sage Pay Flag"]
                social_media_presence_var = ["Snapchat Badge","Pinterest Badge","Facebook Badge","Instagram Badge","Twitter Badge","Linkedin Badge","Yelp Badge","Youtube Badge","Google Badge","Twitter Followers","Twitter Following","Twitter Likes",'Twitter Tweets',"Twitter Post Replies","Twitter Post Retweets","Twitter Post Likes","Insta Followers","Insta Following","Insta Posts","Insta Hashtag Posts"]
                android_itunes_var = ["Android Flag","Android Link","Itunes Flag","Itunes App Link",'Itunes App Flag','Itunes developer Flag','Itunes App Id','Itunes App Title','Itunes App Subtitle','Itunes App Identity','Itunes App Ranking','Itunes App Price','Itunes App Purchase','Itunes App Description','Itunes App Rating','Itunes App Rating Count','Itunes App Seller','Itunes App Size','Itunes App Category','Itunes App Age Rating']
                leadgen_form_var = ["Leadgen Form"]
                eCommerce_keyword_var = ["Ecom Subarch Subscription","Ecom Subarch Pet Food","Ecom Subarch Fast Fashion","Ecom Subarch Jewelry And Accessories","Ecom Subarch Customized Gifts","Ecom Subarch Home Goods","Ecom Subarch Technology","Ecom Subarch Travel","Ecom Subarch Sustainable","Ecom Subarch Beauty","Ecom Subarch Festival And Music Events","Ecom Subarch Video Games","Ecom Subarch Movies And Entertainment","Ecom Subarch Sports And Fitness","Ecom Subarch Toys And Hobbies"]
                
                category_1 = ['pixels','tag_managers','ad_tech','ad_count']
                category_2 = ['cart_checkout','price_points','ecommerce-stack','payments','android_itunes','leadgen-form']
                category_3 = ['social_media_presence']

                # if('social_media_presence' in value):
                #     data_table_2 = result_fin[['Website',"snapchat_badge","pinterest_badge","facebook_badge","instagram_badge","twitter_badge","linkedin_badge","yelp_badge","youtube_badge","google_badge","Twitter_Followers","Twitter_Following","Twitter_Likes",'Twitter_Tweets',"Twitter_Post_Replies","Twitter_Post_Retweets","Twitter_Post_Likes","Insta_Followers","Insta_Following","Insta_Posts","Insta_Hashtag_Posts"]]
                # else:
                #     data_table_2 = pd.DataFrame([[]])

                elem_dict = {'pixels':pixel_var,
                             'tag_managers':tag_manager_var,
                             'ad_tech':ad_tech_var,
                             'ad_count':ad_count_var,
                             'cart_checkout':cart_checkout_var+eCommerce_keyword_var,
                             'price_points':price_point_var,
                             'ecommerce-stack':eCommerce_stack_var,
                             'payments':payment_var,
                             'social_media_presence':social_media_presence_var,
                             'android_itunes':android_itunes_var,
                             'leadgen-form':leadgen_form_var}

                category_1_var = ["Website"]
                for cate in category_1:
                    if cate in value:
                        category_1_var = category_1_var+elem_dict[cate]
                category_2_var = ["Website"]
                for cate in category_2:
                    if cate in value:
                        category_2_var = category_2_var+elem_dict[cate]
                category_3_var = ["Website"]
                for cate in category_3:
                    if cate in value:
                        category_3_var = category_3_var+elem_dict[cate]

                category_4_var = ["Website","Category Score","Price Point Score","Badge Millenium Score","No Pixel Flag","Pixel Score","Ecomm Stack Score","Ecomm Payment Score","Ad Tech Score","Yelp Score","Twitter Page","Insta Page","Influencer Twitter","Influencer Insta","Twitter Engagement Rate","Insta Engagement Rate","Engagement Twitter","Engagement Insta","Twitter Engagement Score","Insta Engagement Score","Influencer Score","Customer Fit Score","Digital Intensity Score","Customer Fit Assign","Digital Intensity Assign","Ecomm Channel Assigned"]
                final_table = channel_assignment(result_fin)

                # final_table = pd.merge(final_table,final_table,on='Website',how='left')

                print(final_table[category_3_var])

                csv_string = final_table.to_csv(index=False, encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
                
                if final_table.empty:
                    return html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],

                        style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-2'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-3'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-4'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in final_table.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-5'),html.Div([html.A('Download Results',download="webCrawledData.csv",id='download-link-1',
                href=csv_string),]),html.Div(html.P("Pixels/Ad-Techs/Tag-Managers/Other",id='table-header-1', className='title',
                style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(
                html.P("ECommerce/App/Lead-Gen-Forms",id='table-header-2',className='title',style={'fontSize':15,'color':'black', 
                    'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Social-Media-Presence",
                    id='table-header-3',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}),
                    className="six columns"),html.Div(html.P("Channel-Assignment",id='table-header-4', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns")
            
                else:
                    return html.Div([dt.DataTable(data=final_table[category_1_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_1_var].columns],

                        style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-2'),html.Div([dt.DataTable(data=final_table[category_2_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_2_var].columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-3'),html.Div([dt.DataTable(data=final_table[category_3_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_3_var].columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-4'),html.Div([dt.DataTable(data=final_table[category_4_var].to_dict('rows'),columns=[{'id': c, 'name': c} for c in final_table[category_4_var].columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-5'),html.Div([html.A('Download Results',download="webCrawledData.csv",id='download-link-1',
                href=csv_string),]),html.Div(html.P("Pixels/Ad-Techs/Tag-Managers/Other",id='table-header-1', className='title',
                style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(
                html.P("ECommerce/App/Lead-Gen-Forms",id='table-header-2',className='title',style={'fontSize':15,'color':'black', 
                    'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Social-Media-Presence",
                    id='table-header-3',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}),
                    className="six columns"),html.Div(html.P("Channel-Assignment",id='table-header-4', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns")
            
        
            # if('pixels' in value and 'tag_managers'):
            #     if('tag_managers' in value):
            #         if('ad_tech' in value):
            #             data_table_3 = result_fin[['Website']+pixel_var+tag_manager_var+ad_tech_var]
            #         else:
            #             data_table_3 = result_fin[['Website']+pixel_var+tag_manager_var]
            #     else:



@app.callback([Output('datatable-6','children'),
                Output('datatable-7','children'),
                Output('datatable-8','children'),
                Output('download-link-2', 'children'),
                Output('table-header-5','children'),
                Output('table-header-6','children'),
                Output('table-header-7','children'),
                ],
                [Input('dropdown-2', 'value'),
                Input('submit-button-1','n_clicks'),
                Input('submit-button-3','n_clicks')],
                [State('upload-data-file', 'contents'),
                State('upload-data-file', 'filename'),
                State('input_button_1', 'value')])
                # [Event('web_elements', 'click')])
def text_crawl(value,n_clicks_1,n_clicks_2,contents,filename,website):
    print(value,n_clicks_1,n_clicks_2,website)
    if n_clicks_2 is not None:
        if contents is not None:
            base = parse_contents(contents, filename)
            print(value)
        # if(n_clicks is not None):
        # if contents is not None:
        # print(rows)
        # base = pd.DataFrame(props['rows'])
        # print(base)
            url_list=[]
            # print(base)
            for url in base['Website']:
                url_list.append(url)
            total_websites = len(url_list)
            counter = 100
            divisor = total_websites//counter
            left_websites = total_websites%counter 
            starting=0
            result=pd.DataFrame()
            for i in range(divisor):
                end=starting +counter
                list_100=url_list[starting:end]
                p =Pool(processes=10)
                rt = p.map(textCrawler, list_100)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = ["Website","Text","Cleaned Text","Non English Flag"]
                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = ["Website","Text","Cleaned Text","Non English Flag"]
                time.sleep(10)
                starting=end
                # file_name = "Web_Crawler_consolidated_03292019_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")
            end=starting + left_websites
            list_remaining=url_list[starting:end]
            p =Pool(processes=10)
            rt = p.map(textCrawler, list_remaining)
            p.terminate()
            p.join()
            lol = [ast.literal_eval(r) for r in rt]
            df = [pd.DataFrame(s) for s in lol]
            res = pd.concat(df)
            if res.empty:
                pass
            else:
                res.columns = ["Website","Text","Cleaned Text","Non English Flag"]

            result = result.append(res)
            if result.empty:
                pass
            else:
                result.columns = ["Website","Text","Cleaned Text","Non English Flag"]
            time.sleep(10)
            starting=end
            # file_name = "Text_Crawler_consolidated_03292019_"+str(end)+".csv"
            # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
            # print ("Done! Please see the csv file")

            # if result.empty:
            #     return html.Div([dt.DataTable(data=[{}])])
            # else:
            #     result.columns = ["Website","Text","Cleaned Text","Non English Flag"]
            #     return html.Div([dt.DataTable(data=result.to_dict('rows'))])

            base = result
            result=pd.DataFrame()
            text_list = []
            web_list = []
            uncleaned_text_list = []
            en_flag_list = []
            for text in base['Cleaned Text']:
                text_list.append(text)
            for web in base['Website']:
                web_list.append(web)
            for uncleaned_text in base['Text']:
                uncleaned_text_list.append(uncleaned_text)
            for en_flag in base['Non English Flag']:
                en_flag_list.append(en_flag)
            total_text = len(text_list)
            counter = 100
            divisor = total_text//counter
            left_text = total_text%counter 
            starting=0
            for i in range(divisor):
                end=starting +counter
                list_100=text_list[starting:end]
                p =Pool(processes=10)
                rt = p.map(classify_update, list_100)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = ["Cleaned Text","Google Api Confidence","Google Api Sub Category","Google Api Category","Google Api Archetype"]
                    result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = ["Cleaned Text","Google Api Confidence","Google Api Sub Category","Google Api Category","Google Api Archetype"]

                time.sleep(10)
                starting=end

                # file_name = "Google_API_Crawler_consolidated_03292019_2.1_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")

            end=starting + left_text
            list_remaining=text_list[starting:end]
            p =Pool(processes=10)
            rt = p.map(classify_update, list_remaining)
            p.terminate()
            p.join()
            lol = [ast.literal_eval(r) for r in rt]
            df = [pd.DataFrame(s) for s in lol]
            res = pd.concat(df)
            if res.empty:
                pass
            else:
                res.columns = ["Cleaned Text","Google Api Confidence","Google Api Sub Category","Google Api Category","Google Api Archetype"]
                result = result.append(res)
            if result.empty:
                pass
            else:
                result.columns = ["Cleaned Text","Google Api Confidence","Google Api Sub Category","Google Api Category","Google Api Archetype"]

            time.sleep(10)
            starting=end

            result['Website'] = web_list
            result['Text'] = uncleaned_text_list
            result['Non English Flag'] = en_flag_list
            # result_fin = pd.merge(base,result,on='Website',how='left')

            result = result[['Website', 'Text', 'Cleaned Text', 'Non English Flag', 'Google Api Confidence','Google Api Sub Category',"Google Api Category","Google Api Archetype"]]

            # file_name = "Google_API_Crawler_consolidated_03292019_2.1_"+str(end)+".csv"
            # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
            # print ("Done! Please see the csv file")

            dataset = result

            nlp = en.load()

            #Step 2: Download and Prepare Stopwords
            #Prerequisites – Download nltk stopwords and spacy model
            #We will need the stopwords from NLTK and spacy’s en model for text pre-processing. 
            #Later, we will be using the spacy model for lemmatization.

            #NLTK Stop words
            stop_words = stopwords.words('english')
            stop_words.extend(['from', 'subject', 're', 'edu', 'use','-', '--', '---', 'a', 'about', 'above', 'across', 
                             'after', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 
                             'always', 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 
                             'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'away', 'b', 
                             'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become', 'becomes', 'been', 
                             'before', 'began', 'behind', 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 
                             'but', 'by', 'c', 'came', 'can', 'cannot', 'case', 'cases', 'certain', 'certainly', 'clear', 
                             'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does', 
                             'done', 'down', 'down', 'downed', 'downing', 'downs', 'during', 'e', 'each', 'early', 'either', 
                             'end', 'ended', 'ending', 'ends', 'enough', 'even', 'evenly', 'ever', 'every', 'everybody', 
                             'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 
                             'few', 'find', 'finds', 'first', 'for', 'four', 'from', 'full', 'fully', 'further', 'furthered', 
                             'furthering', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 
                             'gives', 'go', 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 
                             'grouped', 'grouping', 'groups', 'h', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 
                             'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how', 
                             'however', 'i', 'if', 'important', 'in', 'interest', 'interested', 'interesting', 'interests', 
                             'into', 'is', 'it', 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kind', 'knew', 'know', 
                             'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let',
                             'lets', 'like', 'likely', 'long', 'longer', 'longest', 'm', 'made', 'make', 'making', 'man', 
                             'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most', 'mostly', 'mr', 'mrs', 
                             'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 
                             'new', 'new', 'newer', 'newest', 'next', 'no', 'nobody', 'non', 'noone', 'not', 'nothing', 'now', 
                             'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once',
                             'one', 'only', 'open', 'opened', 'opening', 'opens', 'or', 'order', 'ordered', 'ordering', 
                             'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts',
                             'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing', 'points', 'possible', 
                             'present', 'presented', 'presenting', 'presents', 'problem', 'problems', 'put', 'puts', 'q', 
                             'quite', 'r', 'rather', 'really', 'right', 'right', 'room', 'rooms', 's', 'said', 'same', 'saw',
                             'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several',
                             'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side', 'sides', 'since', 'small', 
                             'smaller', 'smallest', 'so', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state',
                             'states', 'still', 'still', 'such', 'sure', 't', 'take', 'taken', 'than', 'that', 'the', 'their',
                             'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things', 'think', 'thinks', 
                             'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 
                             'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 
                             'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want', 'wanted', 'wanting', 
                             'wants', 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where',
                             'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with', 'within', 'without',
                             'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years', 'yet', 'you', 'young', 
                             'younger', 'youngest', 'your', 'yours', 'z', 'www', 'wwwe', 'com', 'inc', 's', 'uni', 'org'])

            # Step3: Import the document containing scraped data, stored as "ReadText.csv"
            #This is imported using pandas.read_csv and the resulting dataset has 2 columns as shown: Text and Website.

            # dataset = pd.read_csv('Text_crawler_output.csv',encoding='utf-8')
            dataset = dataset.replace(np.nan,'')

            #Step 4. Remove emails and newline characters
            #Let’s get rid of them using regular expressions.

            # Convert to list
            data = dataset["Cleaned Text"].values.tolist()

            # Remove Emails
            data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

            # Remove new line characters
            data = [re.sub('\s+', ' ', sent) for sent in data]

            # Remove distracting single quotes
            data = [re.sub("\'", "", sent) for sent in data]

            #Remove words with less than 3 letters.
            data = [re.sub(r'\W*\b\w{1,3}\b', '', sent) for sent in data]

            dataWeb = dataset.Website.tolist()

            #Step 5: Tokenize words and Clean-up text
            #Let’s tokenize each sentence into a list of words, removing punctuations and unnecessary characters altogether.
            #Gensim’s simple_preprocess() is great for this. Additionally I have set deacc=True to remove the punctuations.

            def sent_to_words(sentences):
              for sentence in sentences:
                  yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

            data_words = list(sent_to_words(data))

            #Step 6: Creating Bigram and Trigram Models
            #Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
            #Some examples in our example are: ‘front_bumper’, ‘oil_leak’, ‘maryland_college_park’ etc.

            #Gensim’s Phrases model can build and implement the bigrams, trigrams, quadgrams and more. 
            #The two important arguments to Phrases are min_count and threshold. 
            #The higher the values of these parameters, the harder it is for words to be combined to bigrams.

            # Build the bigram and trigram models
            bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

            # Faster way to get a sentence clubbed as a trigram/bigram
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            # See trigram example
            print(trigram_mod[bigram_mod[data_words[0]]])

            # See trigram example
            print(trigram_mod[bigram_mod[data_words[2]]])

            #Step 7: Remove Stopwords, Make Bigrams and Lemmatize
            #The bigrams model is ready. 
            #We now need to define the functions to remove the stopwords, make bigrams and lemmatization and call them sequentially.

            # Define functions for stopwords, bigrams, trigrams and lemmatization
            def remove_stopwords(texts):
              return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

            def make_bigrams(texts):
              return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
              return [trigram_mod[bigram_mod[doc]] for doc in texts]

            def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
              """https://spacy.io/api/annotation"""
              texts_out = []
              for sent in texts:
                  doc = nlp(" ".join(sent)) 
                  texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
              return texts_out

            #Let's call the functions in order:
            # Remove Stop Words
            data_words_nostops = remove_stopwords(data_words)

            # Form Bigrams
            data_words_bigrams = make_bigrams(data_words_nostops)

            # Do lemmatization keeping only noun, adj, vb, adv
            data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

            print(data_lemmatized[:1])

            #Step 8: Create the Dictionary and Corpus needed for Topic Modeling
            #The two main inputs to the LDA topic model are the dictionary(id2word) and the corpus. We'll now create them.

            # Create Dictionary
            id2word = corpora.Dictionary(data_lemmatized)

            # Create Corpus
            texts = data_lemmatized

            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]

            # # View
            # print(corpus[:1])

            # Or, you can see a human-readable form of the corpus itself. Human readable format of corpus (term-frequency)
            print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                     id2word=id2word,
                                                     num_topics=10, 
                                                     random_state=100,
                                                     update_every=1,
                                                     chunksize=100,
                                                     passes=10,
                                                     alpha='auto',
                                                     per_word_topics=True)

            pprint(lda_model.print_topics())
            doc_lda = lda_model[corpus]

            #Step 10: Compute Model Perplexity and Coherence Score
            #Model perplexity and topic coherence provide a convenient way to measure how good a given topic model is. 
            #Topic coherence score, in particular, is more helpful.

            # Compute Perplexity
            print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print('\nCoherence Score: ', coherence_lda)

            #Step 11: Visualize the topics-keywords
            #Now that the LDA model is built, the next step is to examine the produced topics and the associated keywords. 
            #We can do that with pyLDAvis package’s interactive chart which is designed to work well with jupyter notebooks.

            # Visualize the topics
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

            pyLDAvis.save_html(vis, 'sample_output.html')

            os.environ['MALLET_HOME'] = 'C:\\Users\\a.daluka\\Documents\\topicM\\mallet-2.0.8'

            #Step 12: Let's improve the coherence score by building Mallet model. 
            #Mallet’s version, however, often gives a better quality of topics.

            # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
            # /Users/aiperiiusupova/Desktop/LEADGEN_ML_CLASSIFICATION/LEADGEN_TOPICMODELING_2/mallet-2.0.8/bin/mallet
            mallet_path ='C:\\Users\\a.daluka\\Documents\\topicM\\mallet-2.0.8\\bin\\mallet'  
            # C:\Users\arunima.prakash\Desktop\mallet-2.0.8\bin
            ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)

            # Step 13: Now, show topics
            pprint(ldamallet.show_topics(formatted=False))

            # Compute Coherence Score
            coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
            coherence_ldamallet = coherence_model_ldamallet.get_coherence()
            print('\nCoherence Score: ', coherence_ldamallet)

            def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
            # """
            # Compute c_v coherence for various number of topics

            # Parameters:
            # ----------
            # dictionary : Gensim dictionary
            # corpus : Gensim corpus
            # texts : List of input texts
            # limit : Max num of topics

            # Returns:
            # -------
            # model_list : List of LDA topic models
            # coherence_values : Coherence values corresponding to the LDA model with respective number of topics
            # """
              coherence_values = []
              model_list = []
              for num_topics in range(start, limit, step):
                  model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
                  model_list.append(model)
                  coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                  coherence_values.append(coherencemodel.get_coherence())

              return model_list, coherence_values

            # Step 14: This can take a long time to run.
            model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

              # Step 15: Show graph
            limit=40
            start=2
            step=6
            x = range(start, limit, step)
            # plt.plot(x, coherence_values)
            # plt.xlabel("Num Topics")
            # plt.ylabel("Coherence score")
            # plt.legend(("coherence_values"), loc='best')
            # plt.show()

            # Step 16: Print the coherence scores
            cv_array = []
            for m, cv in zip(x, coherence_values):
              print("Num Topics =", m, " has Coherence Value of", round(cv, 6))
              cv_array.append(round(cv,6))

            max_index = cv_array.index(max(cv_array))
            print(max_index)
            # Step 17: Select the model and print the topics
            optimal_model = model_list[max_index]
            model_topics = optimal_model.show_topics(formatted=False)
            pprint(optimal_model.print_topics(num_words=15))

            def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data, dataWeb = dataWeb):
              # Init output
              sent_topics_dataset = pd.DataFrame()

              # Get main topic in each document
              for i, row in enumerate(ldamodel[corpus]):
                  row = sorted(row, key=lambda x: (x[1]), reverse=True)
                  # Get the Dominant topic, Perc Contribution and Keywords for each document
                  for j, (topic_num, prop_topic) in enumerate(row):
                      if j == 0:  # => dominant topic
                          wp = ldamodel.show_topic(topic_num)
                          topic_keywords = ", ".join([word for word, prop in wp])
                          sent_topics_dataset = sent_topics_dataset.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                      else:
                          break
              sent_topics_dataset.columns = ['Dominant Topic', 'Perc_Contribution', 'Topic_Keywords']

              # Add original text to the end of the output
              contentsText = pd.Series(texts)
              contentsWebsite = pd.Series(dataWeb)
              sent_topics_dataset = pd.concat([sent_topics_dataset, contentsText, contentsWebsite], axis=1)
              return(sent_topics_dataset)


            dataset_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data, dataWeb = dataWeb)

            # Format
            dataset_dominant_topic = dataset_topic_sents_keywords.reset_index(drop = True)
            dataset_dominant_topic.columns = ['Dominant Topic', 'Topic Perc Contrib', 'Keywords', 'Text', 'Website']

            # # Show
            # dataset_dominant_topic.head(1500)
            dataset_dominant_topic = dataset_dominant_topic[['Website', 'Text', 'Dominant Topic', 'Topic Perc Contrib', 'Keywords']]

            #Step 18: Get the output from the above table. 
            #dataset_dominant_topic.to_csv("output_excel_file.csv", index=False,encoding='utf8')

            dataset_dominant_topic = dataset_dominant_topic.drop(['Text'], axis=1)

            result_fin = pd.merge(dataset,dataset_dominant_topic,on='Website',how='left')

            result_fin.to_csv("output_excel_file_sample_output_03292019.csv", index=False)

            result_fin = result_fin.apply(topic_archetype,axis=1)

            result_fin['Automated Archetype'] = result_fin.apply(auto_archetype,axis=1)

            # result_fin.to_csv("final_archetyping_output_file.csv", index=False)

            result_fin.to_csv("final_archetyping_output_file_03292019.csv", index=False)

            csv_string = result_fin.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

            if result_fin.empty:
                return html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in result_fin.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-6'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in result_fin.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-7'),html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in result_fin.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-8'),html.Div([html.A('Download Results',download="textCrawledData.csv",id='download-link-2',href=csv_string),]),html.Div(html.P("Google-API-Archetype",
                    id='table-header-5',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Topic-Modelling-Archetype",
                    id='table-header-6', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Final-Automated-Archetype",
                    id='table-header-7', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns")
                    
            else:
                return html.Div([dt.DataTable(data=result.to_dict('rows'),id = 'datatable-6',columns=[{'id': c, 'name': c} for c in result.columns],

                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])]),html.Div([dt.DataTable(data=dataset_dominant_topic.to_dict('rows'),id = 'datatable-7',columns=[{'id': c, 'name': c} for c in dataset_dominant_topic.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])]),html.Div([dt.DataTable(data=result_fin.to_dict('rows'),id = 'datatable-8',columns=[{'id': c, 'name': c} for c in result_fin.columns],
                    style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                    style_table={
                        'overflowY': 'scroll',
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])]),html.Div([html.A('Download Results',download="textCrawledData.csv",id='download-link-2',href=csv_string),]),html.Div(html.P("Google-API-Archetype",
                    id='table-header-5',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Topic-Modelling-Archetype",
                    id='table-header-6', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Final-Automated-Archetype",
                    id='table-header-7', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns")
                    
        else:
            if n_clicks_1 is not None:
                url_list = [website]
                total_websites = len(url_list)
                counter = 100
                divisor = total_websites//counter
                left_websites = total_websites%counter 
                starting=0
                result=pd.DataFrame()
                end=starting + left_websites
                list_remaining=url_list[starting:end]
                p =Pool(processes=1)
                rt = p.map(textCrawler, list_remaining)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = ["Website","Text","Cleaned Text","Non English Flag"]

                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = ["Website","Text","Cleaned Text","Non English Flag"]
                time.sleep(10)
                starting=end
                # file_name = "Text_Crawler_consolidated_03292019_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")

                base = result
                result=pd.DataFrame()
                text_list = []
                web_list = []
                uncleaned_text_list = []
                en_flag_list = []
                for text in base['Cleaned Text']:
                    text_list.append(text)
                for web in base['Website']:
                    web_list.append(web)
                for uncleaned_text in base['Text']:
                    uncleaned_text_list.append(uncleaned_text)
                for en_flag in base['Non English Flag']:
                    en_flag_list.append(en_flag)
                total_text = len(text_list)
                counter = 100
                divisor = total_text//counter
                left_text = total_text%counter 
                starting=0
                end=starting + left_text
                list_remaining=text_list[starting:end]
                p =Pool(processes=1)
                rt = p.map(classify_update, list_remaining)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = ["Cleaned Text","Google Api Confidence","Google Api Sub Category","Google Api Category","Google Api Archetype"]
                    result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = ["Cleaned Text","Google Api Confidence","Google Api Sub Category","Google Api Category","Google Api Archetype"]

                time.sleep(10)
                starting=end

                result['Website'] = web_list
                result['Text'] = uncleaned_text_list
                result['Non English Flag'] = en_flag_list
                # result_fin = pd.merge(base,result,on='Website',how='left')

                result = result[['Website', 'Text', 'Cleaned Text', 'Non English Flag', 'Google Api Confidence','Google Api Sub Category',"Google Api Category","Google Api Archetype"]]

                # file_name = "Google_API_Crawler_consolidated_03292019_2.1_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")

                csv_string = result.to_csv(index=False, encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

                if result.empty:
                    return html.Div([dt.DataTable(data=[{}],id='datatable-6',columns=[{'id': c, 'name': c} for c in result.columns],
                        style_as_list_view=True,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                                style_table={
                            'overflowY': 'scroll',
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll'
                        },
                        style_cell={'textAlign': 'center'},
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Website'},
                        'textAlign': 'left'
                        }
                        ])]),html.Div([dt.DataTable(data=[{}],id='datatable-7',columns=[{'id': c, 'name': c} for c in result.columns],
                        style_as_list_view=True,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                            'overflowY': 'scroll',
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll'
                        },
                        style_cell={'textAlign': 'center'},
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Website'},
                        'textAlign': 'left'
                        }
                        ])]),html.Div([dt.DataTable(data=[{}],id='datatable-8',columns=[{'id': c, 'name': c} for c in result.columns],
                        style_as_list_view=True,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                            'overflowY': 'scroll',
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll'
                        },
                        style_cell={'textAlign': 'center'},
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Website'},
                        'textAlign': 'left'
                        }
                        ])]),html.Div([html.A('Download Results',download="textCrawledData.csv",id='download-link-2',href=csv_string),]),html.Div(html.P("Google-API-Archetype",
                        id='table-header-5',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Topic-Modelling-Archetype",
                        id='table-header-6', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Final-Automated-Archetype",
                        id='table-header-7', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns")
                        
                else:
                    dfff = pd.DataFrame()
                    return html.Div([dt.DataTable(data=result.to_dict('rows'),id ='datatable-6',columns=[{'id': c, 'name': c} for c in result.columns],
                        style_as_list_view=True,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                                style_table={
                            'overflowY': 'scroll',
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll'
                        },
                        style_cell={'textAlign': 'center'},
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Website'},
                        'textAlign': 'left'
                        }
                        ])]),html.Div([dt.DataTable(data=[{}],id = 'datatable-7',columns=[{'id': c, 'name': c} for c in dfff.columns],
                        style_as_list_view=True,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                            'overflowY': 'scroll',
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll'
                        },
                        style_cell={'textAlign': 'center'},
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Website'},
                        'textAlign': 'left'
                        }
                        ])]),html.Div([dt.DataTable(data=result.to_dict('rows'),id = 'datatable-8',columns=[{'id': c, 'name': c} for c in result.columns],
                        style_as_list_view=True,
                        style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                            'overflowY': 'scroll',
                            'border': 'thin lightgrey solid',
                            'overflowX': 'scroll'
                        },
                        style_cell={'textAlign': 'center'},
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        },
                        {
                        'if': {'column_id': 'Website'},
                        'textAlign': 'left'
                        }
                        ])]),html.Div([html.A('Download Results',download="textCrawledData.csv",id='download-link-2',href=csv_string),]),html.Div(html.P("Google-API-Archetype",
                        id='table-header-5',className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Topic-Modelling-Archetype",
                        id='table-header-6', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="six columns"),html.Div(html.P("Final-Automated-Archetype",
                        id='table-header-7', className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns")
                    
# Loading screen CSS
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})

@app.callback([Output('datatable-9','children'),
                Output('download-link-3', 'children'),
                Output('table-header-8','children'),
                ],
                [Input('dropdown-3', 'value'),
                Input('submit-button-1','n_clicks'),
                Input('submit-button-4','n_clicks')],
                [State('upload-data-file', 'contents'),
                State('upload-data-file', 'filename'),
                State('input_button_1', 'value')])
                # [Event('web_elements', 'click')])
def image_crawl(value,n_clicks_1,n_clicks_2,contents,filename,website):
    if n_clicks_2 is not None:
        if contents is not None:
            base = parse_contents(contents, filename)
            print(value)

            url_list=[]
            # print(base)
            for url in base['Website']:
                url_list.append(url)
            total_websites = len(url_list)
            counter = 100
            divisor = total_websites//counter
            left_websites = total_websites%counter 
            starting=0
            result=pd.DataFrame()

            col_list = ["Website","Image Name","Object Label","Object Probabiliy"]

            for i in range(divisor):
                end=starting +counter
                list_100=url_list[starting:end]
                p =Pool(processes=2)
                rt = p.map(image_type, list_100)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = col_list
                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = col_list

                result.drop_duplicates(keep='first', inplace=True)

                time.sleep(10)
                starting=end

                # file_name = "Image_Crawler_consolidated_04082019_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")

            end=starting + left_websites
            list_remaining=url_list[starting:end]
            p =Pool(processes=2)
            rt = p.map(image_type, list_remaining)
            p.terminate()
            p.join()
            lol = [ast.literal_eval(r) for r in rt]
            df = [pd.DataFrame(s) for s in lol]
            res = pd.concat(df)
            if res.empty:
                pass
            else:
                res.columns = col_list

            result = result.append(res)
            if result.empty:
                pass
            else:
                result.columns = col_list

            result.drop_duplicates(keep='first', inplace=True)

            time.sleep(10)
            starting=end

            csv_string = result.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
            
            if result.empty:
                return html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in result.columns],

                    style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                                 style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-9'),html.Div([html.A('Download Results',download="imageCrawledData.csv",id='download-link-3',href=csv_string),]),html.Div(html.P("Image-Analytics",
                id = "table-header-8",className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns"),
                
            else:
                return html.Div([dt.DataTable(data=result.to_dict('rows'),columns=[{'id': c, 'name': c} for c in result.columns],

                    style_as_list_view=True,
                style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                                 style_table={
                    'overflowY': 'scroll',
                    'height':450,
                    'border': 'thin lightgrey solid',
                    'overflowX': 'scroll'
                },
                style_cell={'textAlign': 'center'},
                style_cell_conditional=[{
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                },
                {
                'if': {'column_id': 'Website'},
                'textAlign': 'left'
                }
                ])],id='datatable-9'),html.Div([html.A('Download Results',download="imageCrawledData.csv",id='download-link-3',href=csv_string),]),html.Div(html.P("Image-Analytics",
                id = "table-header-8",className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns"),
        else:
            if n_clicks_1 is not None:
                url_list = [website]
                total_websites = len(url_list)
                counter = 100
                divisor = total_websites//counter
                left_websites = total_websites%counter 
                starting=0
                result=pd.DataFrame()
                col_list = ["Website","Image Name","Object Label","Object Probabiliy"]
                for i in range(divisor):
                    end=starting +counter
                    list_100=url_list[starting:end]
                    p =Pool(processes=1)
                    rt = p.map(image_type, list_100)
                    p.terminate()
                    p.join()
                    lol = [ast.literal_eval(r) for r in rt]
                    df = [pd.DataFrame(s) for s in lol]
                    res = pd.concat(df)
                    if res.empty:
                        pass
                    else:
                        res.columns = col_list
                    result = result.append(res)
                    if result.empty:
                        pass
                    else:
                        result.columns = col_list

                    result.drop_duplicates(keep='first', inplace=True)

                    time.sleep(10)
                    starting=end

                    # file_name = "Image_Crawler_consolidated_04082019_"+str(end)+".csv"
                    # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                    # print ("Done! Please see the csv file")

                end=starting + left_websites
                list_remaining=url_list[starting:end]
                p =Pool(processes=1)
                rt = p.map(image_type, list_remaining)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = col_list

                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = col_list

                result.drop_duplicates(keep='first', inplace=True)

                time.sleep(10)
                starting=end

                # final_table = pd.merge(final_table,final_table,on='Website',how='left')

                # print(final_table[category_3_var])

                csv_string = result.to_csv(index=False, encoding='utf-8')
                csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
                
                if result.empty:
                    return html.Div([dt.DataTable(data=[{}],columns=[{'id': c, 'name': c} for c in result.columns],

                        style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                        'overflowY': 'scroll',
                        # 'height':450,
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-9'),html.Div([html.A('Download Results',download="imageCrawledData.csv",id='download-link-3',href=csv_string),]),html.Div(html.P("Image-Analytics",
                    id = "table-header-8",className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns"),
                else:
                    return html.Div([dt.DataTable(data=result.to_dict('rows'),columns=[{'id': c, 'name': c} for c in result.columns],

                        style_as_list_view=True,
                    style_header={'backgroundColor': 'rgb(30, 30, 30)','color':'white'},
                        style_table={
                        'overflowY': 'scroll',
                        'height':450,
                        'border': 'thin lightgrey solid',
                        'overflowX': 'scroll'
                    },
                    style_cell={'textAlign': 'center'},
                    style_cell_conditional=[{
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    },
                    {
                    'if': {'column_id': 'Website'},
                    'textAlign': 'left'
                    }
                    ])],id='datatable-9'),html.Div([html.A('Download Results',download="imageCrawledData.csv",id='download-link-3',href=csv_string),]),html.Div(html.P("Image-Analytics",
                    id = "table-header-8",className='title',style={'fontSize':15,'color':'black', 'fontWeight':'bold',"textAlign":"center"}), className="twelve columns"),

@app.callback([Output('lead_div_1','children'),
                Output('lead_div_2', 'children')
                ],
                [Input('submit-button-L1','n_clicks')],
                [State('upload-data-file-L1', 'contents'),
                State('upload-data-file-L1', 'filename')])
                # [Event('web_elements', 'click')])
def lead_search_engine(n_clicks,contents,filename):
    if n_clicks is not None:
        # print(n_clicks)
        if contents is not None:
            base = parse_contents(contents, filename)
            url_list=[]
            # print(base)
            for url in base['Website']:
                url_list.append(url)
            total_websites = len(url_list)
            counter = 100
            divisor = total_websites//counter
            left_websites = total_websites%counter 
            starting=0
            result=pd.DataFrame()
            for i in range(divisor):
                end=starting +counter
                list_100=url_list[starting:end]
                p =Pool(processes=10)
                rt = p.map(textCrawler, list_100)
                p.terminate()
                p.join()
                lol = [ast.literal_eval(r) for r in rt]
                df = [pd.DataFrame(s) for s in lol]
                res = pd.concat(df)
                if res.empty:
                    pass
                else:
                    res.columns = ["Website","Text","Cleaned Text","Non English Flag"]
                result = result.append(res)
                if result.empty:
                    pass
                else:
                    result.columns = ["Website","Text","Cleaned Text","Non English Flag"]
                time.sleep(10)
                starting=end
                # file_name = "Web_Crawler_consolidated_03292019_"+str(end)+".csv"
                # result.to_csv(file_name, sep=',', index = False, encoding='utf-8')
                # print ("Done! Please see the csv file")
            end=starting + left_websites
            list_remaining=url_list[starting:end]
            p =Pool(processes=10)
            rt = p.map(textCrawler, list_remaining)
            p.terminate()
            p.join()
            lol = [ast.literal_eval(r) for r in rt]
            df = [pd.DataFrame(s) for s in lol]
            res = pd.concat(df)
            if res.empty:
                pass
            else:
                res.columns = ["Website","Text","Cleaned Text","Non English Flag"]

            result = result.append(res)
            if result.empty:
                pass
            else:
                result.columns = ["Website","Text","Cleaned Text","Non English Flag"]
            time.sleep(10)
            starting=end



            return html.Div(dcc.Input(
                    id = 'input_button_L1',
                    placeholder='Enter query to search across web pages...',
                    type='text',
                    value='',
                    style={'width': '100%'}
                    ),className="eleven columns"),html.Div(html.Button(id='submit-button-L2', type='submit', children='Submit',style={'color':'white','backgroundColor':'#506784'}),style={'float':'right'})

                
# Loading screen CSS
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

if __name__ == '__main__':
    app.server.run()
