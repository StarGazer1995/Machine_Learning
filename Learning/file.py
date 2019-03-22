import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd


filename=os.path.abspath("G:\\Kaggle\\Data Set\\all\\train_labels.csv")
data=pd.read_csv(filename)
