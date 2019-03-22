from numpy import *
from os import listdir
from math import *
import operator

def calShannonEnt(dataSet):
        numEntries=len(dataSet)
        labelCounts={}
        for featVec in dataSet:
                currentLabel= featVec[-1] #每行数据的最后一个字（类别）
                if currentLabel not in labelCounts.keys():
                        labelCounts[currentLabel]=0
                labelCounts[currentLabel]+=1
        shannonEnt=0.0
        for key in labelCounts:
                prob=float(labelCounts[key])/numEntries
                shannonEnt -= prob*math.log(prob,2)
        return shannonEnt

def creatDataSet():
        DataSet=[     [1,1,'Yes'],
                      [1,0,'No' ],
                      [0,1,'No' ],
                      [0,1,'No' ],
                      [1,1,'Yes']]
        labels=['no surfacing','flippers']
        return DataSet,labels

def splitDataSet(dataSet,axis,value):
        retDataSet = []
        for featVec in dataSet:
                if featVec[axis]==value:
                        reducedFeatVec=featVec[:axis]#输入在axis之前的信息
                        reducedFeatVec.extend(featVec[axis+1:])#输入axis之后的信息
                        retDataSet.append(reducedFeatVec)
        return retDataSet

def chooseBestFeatureToSplit(dataSet):
        numfeature=len(dataSet[0])-1
        baseEntropy=calShannonEnt(dataSet)
        bestInforGain=0.0;bestFeature=2
        for i in range(numfeature):
                featList=[example[i] for example in dataSet]
                uniqueVals=set(featList)
                newEntropy=0.0
                for value in uniqueVals:
                        subDataSets=splitDataSet(dataSet,i,value)
                        prob=len(subDataSets)/float(len(dataSet))
                        newEntropy +=prob*calShannonEnt(subDataSets)
                infoGain= baseEntropy-newEntropy
                if (infoGain>bestInforGain):
                        bestEntropy=infoGain
                        bestFeature= i
        return bestFeature
def majorityCnt(classList):
        classCount={}
        for vote in ClassList:
                if vote not in classCount.keys():
                        classCount[vote]=0
                classCount+=1
                sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

def createTree(dataSet,labels):
        classList=[example[-1] for example in dataSet]
        if classList.count(classList[0])==len(classList):
                return classList[0]
        if len(dataSet[0])==1:
                return majorityCnt(classList)
        bestFeat=chooseBestFeatureToSplit(dataSet)
        bestFeatLabel=labels[bestFeat]
        myTree={bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues=[example[bestFeat] for example in dataSet]
        uniqueVals=set(featValues)
        for value in uniqueVals:
                subLabels=labels[:]
                myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
        return myTree


def getNumleafs(mytree):
    numleafs=0
    firstStr=list(mytree.keys())[0]
    secondDict=mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numleafs+=getNumleafs(secondDict[key])
        else: numleafs+=1
    return numleafs

def getTreeDepth(mytree):
    maxDepth=0
    firstStr=list(mytree.keys())[0]
    secondDict=mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                {'no surfacing':{0:'no',1:{'flippers':{0:{'heads':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]