import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import heapq
import matplotlib.pyplot as plt
import sys
eps = np.finfo(float).eps

#parameters 
classIndex = 7 # index of the output variable
features = [0,1,2,3,4,5,6] # index of the independent variables


pdata = None
try:
    pdata = pd.read_csv(sys.argv[1])
except:
    dataloc = str(input("Enter the location of training data "))
    pdata = pd.read_csv(dataloc)

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

data = pdata.values[:,1:]
np.random.shuffle(data)

split = 0.8

def getCoeff(X,Y):
    X = np.hstack((np.ones([X.shape[0],1]),X))
    C = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),Y)
    return C

def MAE(ya,yp):
    return abs(ya - yp)
def MSE(ya,yp):
    return (ya - yp)**2
def MPE(ya,yp):
    return 100 * ((ya - yp)/ya)

def LRPredict(coeff,test):
    i = 0
    r = coeff[0].copy()
    while i < test.shape[0]:
        r += (coeff[i+1] * test[i])
        i += 1
    return r

def trainTest(data,classIndex,features,t=0.8,coeff=None,errofunc=MSE,test=None):
    split = int(t*data.shape[0])
    train = data[0:split]
    vald = data[split:]
    
    coeff = getCoeff(train[:,features],train[:,classIndex].reshape(-1,1)).T[0]
    #if coeff is None:
    #    coeff = trainLinearRegression(train[:,features],train[:,classIndex].reshape(-1,1),func=leastSqDiff,th=0.002)
    
    lr = linear_model.LinearRegression()
    lr.fit(train[:,features],train[:,classIndex])
    #print("My coeff",coeff,"\nScikit coeff",lr.intercept_,lr.coef_)
    
    diff = 0
    for i in vald:
        diff += errofunc(i[classIndex],LRPredict(coeff,i[features]))
    
    if test is not None:
        print("Test data")
        for i in test:
        	print("Data",i,"Prediction",LRPredict(coeff,i[features]))
    return coeff,diff/vald.shape[0]
split = 0.8
coeff,mse = trainTest(data,classIndex,features,t = 0.8,test=tdata)
coeff,mae = trainTest(data,classIndex,features,t = 0.8,coeff=coeff,errofunc=MAE)
coeff,mpe = trainTest(data,classIndex,features,t = 0.8,coeff=coeff,errofunc=MPE)

print("Mean Sqare Test Error",mse, \
     "\nMean Absolute Error",mae,\
     "\nMean Percentage Error",mpe)
print("Coefficients")
for i,j in list(zip(pdata.columns.values[1:],coeff[1:])):
    print(i,j)
print("Intercept",coeff[0])

#Plots on 'split' percentage of data
data2 = data[0:int(split * data.shape[0])]
c,m = getCoeff(data2[:,0].reshape(-1,1),data2[:,classIndex].reshape(-1,1))
x = np.arange(np.min(data2[:,0]) - 20, np.max(data2[:,0]) + 20)
y = m*x + c
yp = coeff[1]*x
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(data2[:,0],data2[:,classIndex],s = 2)
plt.plot(x,y,c="red",label="Only GRE")
plt.plot(x,yp,c="green",label="ALL Atributes")
plt.title("GRE")
plt.legend()

plt.subplot(1,3,2)
c,m = getCoeff(data2[:,5].reshape(-1,1),data2[:,classIndex].reshape(-1,1))
x = np.arange(np.min(data2[:,5]) - 1, np.max(data2[:,5]) + 1)
y = m*x + c
plt.scatter(data2[:,5],data2[:,classIndex],s = 2)
plt.plot(x,y,c="red",label="Only CGPA")
yp = coeff[6]*x -0.3
plt.plot(x,yp,c="green",label="All attributes")
plt.title("CGPA")
plt.legend()

plt.subplot(1,3,3)
c,m = getCoeff(data2[:,3].reshape(-1,1),data2[:,classIndex].reshape(-1,1))
x = np.arange(np.min(data2[:,3]) - 1, np.max(data2[:,3]) + 2)
y = m*x + c
plt.scatter(data2[:,3],data2[:,classIndex],s = 2)
plt.plot(x,y,c="red",label="Only SOP")
yp = coeff[6]*x + 0.3
plt.plot(x,yp,c="green",label="All attributes")
plt.title("SOP")
plt.legend()
plt.show()

