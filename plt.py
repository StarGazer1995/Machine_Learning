import matplotlib.pyplot as plt
import decisiontree as dec
decisionNode= dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeText,centerPt,parentPt,nodeType):
    createPlot.axl.annotate(nodeText,xy=parentPt,xytext=centerPt,xycoords='axes fraction',
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
    
def plotMidText(cntrPt,ParentPt,txtString):
        xMid=(ParentPt[0]-cntrPt[0])/2.0+cntrPt[0]
        yMid=(ParentPt[1]-cntrPt[0])/2.0+cntrPt[1]
        createPlot.axl.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodePt):
        numLeafs=dec.getNumleafs(myTree)
        dec.getTreeDepth(myTree)#意义不明的一行
        firstStr=list(myTree.keys())[0]
        cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
        plotMidText(cntrPt,parentPt,nodePt)
        plotNode(firstStr,cntrPt,parentPt,decisionNode)
        secondDict=myTree[firstStr]
        plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
        for key in secondDict.keys():
                if type(secondDict[key]).__name__=='dict':
                        plotTree(secondDict[key],cntrPt,str(key))
                else:
                        plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
                        plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
                        plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
        plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD

def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.axl=plt.subplot(111,frameon=False, **axprops)
    plotTree.totalW=float(dec.getNumleafs(inTree))
    plotTree.totalD=float(dec.getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW;plotTree.yOff=1.0
    plotTree(inTree,(0.5,0.1),'')
    plt.show()
