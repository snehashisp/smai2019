import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import heapq
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
import sys


#parmeters for iris dataset
#it is assumed that all train and test data are in the format provided(including preceeding spaces)
classIndex = 4 # this is the column index of the label to be predicted
features = [0,1,2,3] # these are column indexes of the set of features use to predict the label

#load data from command line arguments
data = None
try:
    data = pd.read_csv(sys.argv[1],header=None).values
except:
    dataloc = str(input("Enter the location of training data "))
    data = pd.read_csv(dataloc,header=None).values

tdata = None
try:
    tdata = pd.read_csv(sys.argv[2],header=None).values
except:
    tdataloc = str(input("Enter the location of test data (leave blank for no test data) "))
    if tdataloc == "":
        tdata = None
    else:
        tdata = pd.read_csv(tdataloc,header=None).values


np.random.shuffle(data)
#distance parameters
def norm(p):
    
    def pnorm(point,newPoint):
        euNorm = 0
        for i,j in list(zip(point,newPoint)):
            euNorm += abs(i - j)**p
        if p == 0:
            return euNorm
        return euNorm**(1/float(p))
    return pnorm

#find k minimum data points
def kminDist(pointSet,point,func,k):
    dist = [(func(point,i),j) for i,j in list(zip(pointSet,range(len(pointSet))))]
    #print(dist)
    return heapq.nsmallest(k,dist,key = lambda x: int(x[0]))

def sciKnn(dataset,classIndex,features,k,t):
    split = int(t*dataset.shape[0])
    train = dataset[0:split,features]
    tlabel = dataset[0:split,classIndex]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train,tlabel.astype(str))
    validate = dataset[split:,features]
    vlabel = dataset[split:,classIndex]
    return knn.score(validate,vlabel.astype(str))


#k-means classifier

#knn predictor features is the n dimensional feature indices
def knnPredict(dataset,classIndex,features,k,testdata,func,labels):
    
    cdict = {}
    for c in labels:
        cdict[c] = np.array([0,float(0)])
    distVec = kminDist(dataset[:,features],testdata,func,k)
    for j in distVec:
        cdict[dataset[j[1]][classIndex]] += (1,j[0])
    
    vdist = [(i[0],i[1],j) for i,j in list(zip(cdict.values(),cdict.keys()))]
    vdist.sort(key = lambda x:(x[0],-x[0]),reverse=True)
    #print(vdist)
    return vdist[0][2]
    
#splits dataset into training and test(in this case an initial amount of data is not used for prediction)
def cknnTest(dataset,classIndex,features,k,t=0.9,func=norm(2),test=None,showstats=None):
    
    split = int(t*dataset.shape[0])
    pointSet = dataset[0:split]
    labels = np.unique(dataset[:,classIndex])
    
    confMat = {}
    for i in labels:
        confMat[i] = {}
        for j in labels:
            confMat[i][j] = 0
        
    for i in dataset[split:]:
        #print(i)
        res = knnPredict(pointSet,classIndex,features,k,i[features],func,labels)
        confMat[i[classIndex]][res] += 1
    
    if test is not None:
        print("Test Data")
        for i in test:
            print("Data",i,"Predicted Label",knnPredict(pointSet,classIndex,features,k,i[features],func,labels)) 

    a = 0
    for i in labels:
        a += confMat[i][i]
        if showstats is None:
            continue
        print("Label ",i)
        r = 0
        for k in labels:
            r += confMat[i][k]
        print("Recall",confMat[i][i]/r)
        p = 0
        for k in labels:
            p += confMat[k][i]
        print("Precision",confMat[i][i]/p)
        print("---------------")
        
            
    return a/dataset[split:].shape[0]

print("Iris Dataset \n--------------------------------")
print("k = 7 80% train 20% test Using euclidean distance")
print("My Implementation")
print("Overall Accuracy",cknnTest(data,classIndex,features,k=7,t=0.8,func=norm(2),test=tdata,showstats=1))
print("Scikit Accuracy")
print(sciKnn(data,classIndex,features,k=7,t=0.8))

x,y1,y2 = [],[],[]
for i in range(1,20,2):
    x += [i]
    y1 += [cknnTest(data,classIndex,features,i,t=0.8)]
    y2 += [cknnTest(data,classIndex,features,i,t=0.8,func=norm(1))]


plt.plot(x,y1)
plt.plot(x,y2)
plt.legend(["Eucludean","Manhattan"])
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()