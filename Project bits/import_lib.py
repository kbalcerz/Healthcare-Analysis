from IPython import get_ipython
ipython = get_ipython()
ipython.magic("load_ext google.cloud.bigquery")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from google.cloud import storage
import seaborn as sns
import os, sys
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=False)
sys.path.append('../Downloads')
import ML_analysis

