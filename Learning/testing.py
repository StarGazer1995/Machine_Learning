from numpy import *
from os import listdir
from math import *
import matplotlib.pyplot as plt
import operator

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distance=sqDistance**0.5
    sortedDistIndicies=distance.argsort()
    classCount={}
    for i in range(k):
            voteIlabel=labels[sortedDistIndicies[i]]
            classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCount[0] [0]
    
def aotoNorm(dataSet):
    minVals= dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zores(shape(dataSet))
    m= dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=dataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat, datingLabels=file2matrix('dataTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m= normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult= classify0(normMat[i,:])



myDat,label=creatDataSet()
mytree=createTree(myDat,label)
print(mytree)