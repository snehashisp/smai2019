import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import heapq
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
import sys



#load data from command line arguments
data = None
try:
    data = pd.read_csv(sys.argv[1]).values
except:
    dataloc = str(input("Enter the location of training data "))
    data = pd.read_csv(dataloc).values

tdata = None
try:
    tdata = pd.read_csv(sys.argv[2]).values
except:
    tdataloc = str(input("Enter the location of test data (leave blank for no test data) "))
    if tdataloc == "":
        tdata = None
    else:
        tdata = pd.read_csv(tdataloc).values
if tdata is not None:
    tdata = tdata[:,1:]
    
data = data[:,1:]
np.random.shuffle(data)

#parameters
#dict having boolean values determining which type is pirticular attribute(feature)
#0 categorical , 1 numerical
featureType = {}
featureType[0] = 1 #age
featureType[1] = 1 #number of years of exp
featureType[2] = 1 #annual income
#featureType[3] = 1 #ZIP Code
featureType[4] = 1 #family size
featureType[5] = 1 #average spending
featureType[6] = 0 #education level
featureType[7] = 1 #mortgage
featureType[9] = 0 #securities
featureType[10] = 0 #CD
featureType[11] = 0 #internet banking
featureType[12] = 0 #credit card
#columns(attribute) indices of features
features = list(featureType.keys())
#column index of label
classIndex = 8


def condProbCat(data_label):
    d = {}
    ulabel = np.unique(data_label[:,1])
    for l in ulabel:
        du = data_label[data_label[:,1] == l,0]
        total = du.shape[0]
        udu = np.unique(du,return_counts=True)
        for i,j in list(zip(udu[0],udu[1])):
            d[(l,i)] = j/total
    #print(d)      
    def retFunc(val,cond):
        if (cond,val) in d.keys():
            return d[(cond,val)]
        return -1
    return retFunc

def normalDist(p,mean,sdev):
    return (1/(((2*np.pi)**(0.5))*sdev)) * np.e**(-0.5*(((p - mean)**2)/(sdev**2)))

def condProbNum(data_label):
    d = {}
    ulabel = np.unique(data_label[:,1])
    for l in ulabel:
        du = data_label[data_label[:,1] == l,0]
        d[l] = (np.mean(du),np.std(du))
    #print(d)   
    def retFunc(val,cond):
        return normalDist(val,d[cond][0],d[cond][1])
    return retFunc

def createBayes(data,classIndex,features,featureType):
    
    bayesfunc = {}
    for i in features:
        if featureType[i] == 0:
            bayesfunc[i] = condProbCat(data[:,[i,classIndex]])
        else:
            bayesfunc[i] = condProbNum(data[:,[i,classIndex]])
            
    bayesPrior = {}
    ulabel = np.unique(data[:,classIndex],return_counts = True)
    for i,j in list(zip(ulabel[0],ulabel[1])):
        bayesPrior[i] = j/data.shape[0]
    
    return (bayesfunc,bayesPrior)

def bayesPredict(bayes,test,features):
    
    func,prior = bayes[0],bayes[1]
    label = list(prior.keys())[0]
    maxProb = 0
    for i in list(prior.keys()):
        prob = prior[i]
        for j in features:
            p = func[j](test[j],i)
            if p <= 0:
                p = prior[i]
            prob *= p
        if prob > maxProb:
            maxProb = prob
            label = i
    return label
\
def bayesTest(data,classIndex,features,featureType,t=0.8,test=None):
    split = int(t * data.shape[0])
    train = data[0:split]
    validate = data[split:]
    bayes = createBayes(train,classIndex,features,featureType)

    t,f = 0,0
    for i in validate:
        pl = bayesPredict(bayes,i,features)
        if pl == i[classIndex]:
            t += 1
        else:
            f += 1
    
    if test is not None:
        print("Test data")
        for i in test:
            print("Data",i,"Prediction",bayesPredict(bayes,i,features))
            
    print("My Accuracy",t/(t+f))
    clf = GaussianNB()
    clf.fit(train[:,features],train[:,classIndex])
    print("Scikit",clf.score(validate[:,features],validate[:,classIndex]))
    

bayesTest(data,classIndex,features,featureType,t=0.8,test=tdata)