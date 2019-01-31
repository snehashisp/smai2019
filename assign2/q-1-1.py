import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import heapq
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
import sys


#parmeters for robot dataset
#it is assumed that all train and test data are in the format provided(including preceeding spaces)
classIndex = 0 # this is the column index of the label to be predicted
features = [1,2,3,4,5,6] # these are column indexes of the set of features use to predict the label

#load data from command line arguments
data = None
try:
    data = pd.read_csv(sys.argv[1],sep=" ",header=None).values[:,1:-1]
except:
    dataloc = str(input("Enter the location of training data "))
    data = pd.read_csv(dataloc,sep=" ",header=None).values[:,1:-1]

tdata = None
try:
    tdata = pd.read_csv(sys.argv[2],sep=" ",header=None).values
except:
    tdataloc = str(input("Enter the location of test data (leave blank for no test data) "))
    if tdataloc == "":
        tdata = None
    else:
        tdata = pd.read_csv(tdataloc,sep=" ",header=None).values

if tdata is not None:
    tdata = tdata[:,1:-1]

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
    return heapq.nsmallest(k,dist,key = lambda x: x[0])

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
def knnTest(dataset,classIndex,features,k,t=0.9,func=norm(2),test=None):
    
    split = int(t*dataset.shape[0])
    pointSet = dataset[0:split]
    labels = np.unique(dataset[:,classIndex])
    
    tp,tn,fp,fn = 0,0,0,0
    for i in dataset[split:]:
        #print(i)
        res = knnPredict(pointSet,classIndex,features,k,i[features],func,labels)
        if res == i[classIndex]:
            if res == 1:
                tp += 1
            else:
                tn += 1
        else:
            if res == 1:
                fp += 1
            else:
                fn += 1
    if test is not None:
        print("Test Data")
        for i in test:
            print(i,"Predicted Label",knnPredict(pointSet,classIndex,features,k,i[features],func,labels))
            
    return (tp,tn,fp,fn),((tp + tn)/(tp + tn + fp + fn))


print("My Implementation using eucludean distance 80% training data k = 7")
tp,tn,fp,fn = knnTest(data,classIndex,features,k=7,t=0.8,func=norm(2),test=tdata)[0]
print("Accuracy",(tp + tn)/(tp + tn + fp + fn),"Recall",tp/(tp + fn), \
      "Precision",tp/(tp + fp),"F1 score",2*tp/(2*tp + fp + fn))
print("Scikit Accuracy")
print(sciKnn(data,classIndex,features,k=7,t=0.8))

x,y1,y2 = [],[],[]
for i in range(1,18,2):
    x += [i]
    y1 += [knnTest(data,classIndex,features,i,t=0.8)[1]]
    y2 += [knnTest(data,classIndex,features,i,t=0.8,func=norm(1))[1]]


plt.plot(x,y1)
plt.plot(x,y2)
plt.legend(["Eucludean","Manhattan"])
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()