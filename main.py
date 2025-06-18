import warnings
import h2o
from h2o import H2OFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging


h2o.init(max_mem_size='4G')
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

logging.info('Importing all the necessary .cvs files (10-K reports)...')
#importing .cvs file for all companies within one sector

###

#STAGE 1: DATA INSPECTION
logging.info('Starting data inspection...')



