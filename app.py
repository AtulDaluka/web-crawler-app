import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
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
########### Set up the chart
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
