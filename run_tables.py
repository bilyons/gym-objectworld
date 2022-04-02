import random
import numpy as np
import os
import sys
from scipy.special import softmax
from scipy.stats import pearsonr
import pandas as pd



df = pd.read_csv(os.path.abspath(os.getcwd())+'/data/0/vairl.csv')
print(df)