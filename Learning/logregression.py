from numpy import *

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMat,classLabels):
    dataMatrix=mat(dataMat)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.01;cycle_times=500
    weight=ones((n,1))
    for k in cycle_times:
        h=sigmoid(dataMatrix*weight)
        error=(labelMat-h)
        weight=weight+alpha*dataMatrix.transpose()*error
    return weight

def gradDescent(dataMat,classLabels):
    dataMatrix=mat(dataMat)
    labelMat=mat(classLabels)
    m,n=shape(dataMatrix)
    alpha=0.01;cycle_times=500
    weight=ones((n,1))
    for k in cycle_times:
        h=sigmoid(dataMatrix*weight)
        error=(labelMat-h)
        weight=weight-alpha*dataMatrix.transpose()*error
    return weight
