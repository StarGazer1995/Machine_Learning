from numpy import *
def loadDataSet():
    postingList=[['my','dog','has','flea','problem','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['qiut','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def creatVocablist(dataSet):
    vacabSet=set([])
    for document in dataSet:
        vacabSet=vacabSet | set(document)
    return list(vacabSet)

def setOfWords2Vec(vacabList,inputSet):
    returnVec=[0]*len(vacabList)
    for word in inputSet:
        if word in vacabList:
            returnVec[vacabList.index(word)]=1
        else: print("the word: %s is not in the vocabulary" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords); p1Num=ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom=sum(trainMatrix[i])
    p1Vect= p1Num/p1Denom;p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+(1-log(pClass1))
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myvocablist=creatVocablist(listOPosts)
    trainMat=[]
    for postDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myvocablist,postDoc))
    p0,p1,pa=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dakmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    