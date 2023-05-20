import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = 'C:\\abdelrahman\\classification.txt'
data= pd.read_csv(path,header=None,names=['exam1','exam2','admitted'])
print('data = ',data.head(10))
print('data describe = ',data.describe())

positive=data[data['admitted'].isin([1])]
negative=data[data['admitted'].isin([0])]
fig,ax= plt.subplots(figsize=(8,5))
ax.scatter(positive['exam1'],positive['exam2'],s=50,c='b',marker='o',label='admitted')
ax.scatter(negative['exam1'],negative['exam2'],s=50,c='r',marker='x',label='not admitted')
ax.legend()
ax.set_xlabel('exam1 score')
ax.set_ylabel('exam2 score')

def sigmoid(z):
    return 1/(1+ np.exp(-z))

nums=np.arange(-10,10,step =1)


fig,ax=plt.subplots(figsize=(8,5))
ax.plot(nums,sigmoid(nums),'r')

data.insert(0,'ones',1)
cols=data.shape[1]
X=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
X=np.array(X.values)
y=np.array(y.values)
theta=np.zeros(3)
def cost(thetav,xv,yv):
    thetav=np.matrix(thetav)
    xv=np.matrix(xv)
    yv=np.matrix(yv)
    first=np.multiply(-yv,np.log(sigmoid(xv * thetav.T)))
    second=np.multiply((1 -yv),np.log(1-sigmoid(xv * thetav.T)))
    return np.sum(first - second) / (len(xv))

thiscost= cost(theta,X,y)
print('cost = ',thiscost)
def gradient(thetav,xv,yv):
    thetav=np.matrix(thetav)
    xv=np.matrix(xv)
    yv=np.matrix(yv)
    parametrs=int(thetav.ravel().shape[1])
    grad=np.zeros(parametrs)
    error=sigmoid(xv * thetav.T)-yv
    for i in range(parametrs):
        term=np.multiply(error,xv[:,1])
        grad[i]=np.sum(term)/len(xv)
    return grad

import scipy.optimize as opt
result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y))
print('result=',result)
costafter=cost(result[0],X,y)
print('cost after =',costafter)
def predict(theta,X):
    probability=sigmoid(X*theta.T)
    return[1 if x>=0.5 else 0 for x in probability]
theta_min=np.matrix(result[0])
predicition=predict(theta_min,X)
correct=[1 if((a==1 and b==1) or (a==0 and b==0)) else 0 for(a,b) in zip(predicition,y)]
accuracy=(sum(map(int,correct)) % len(correct))
print('accuracy = ', format(accuracy))


    


