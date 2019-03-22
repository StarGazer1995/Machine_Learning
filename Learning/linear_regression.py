import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def createFunction():
    beta0=2;beta1=3
    stdDeviation=1;ErrMean=0;numberSamples=100
    error=np.random.normal(loc=ErrMean,scale=stdDeviation,size=numberSamples)
    x=np.linspace(-2,2,numberSamples)
    y=beta0+beta1*x+error
    y=y.reshape(-1,1);x=x.reshape(-1,1)
    return x,y
