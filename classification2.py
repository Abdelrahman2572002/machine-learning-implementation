import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
path = 'C:\\abdelrahman\\c.txt'
data= pd.read_csv(path,header=None,names=['test1','test2','accepted'])
print('data = ',data.head(10))
print('data describe = ',data.describe())
positive=data[data['accepted'].isin([1])]
negative=data[data['accepted'].isin([0])]
fig,ax=plt.subplots(figsize=(5,5))
ax.scatter(positive['test1'],positive['test2'],s=50,c='b',marker='o',label='accepted')
ax.scatter(negative['test1'],negative['test2'],s=50,c='r',marker='x',label='rejected')
ax.legend()
ax.set_xlabel('tes1 score')
ax.set_ylabel('tes2 score')

degree=5
x1=data['test1']
x2=data['test2']
data.insert(3,'ones',1)
for i in range(1,degree):
    for j in range(0,i):
        data['F'+str(i)+str(j)]=np.power(x1,i-j) *np.power(x2,j)
        
data.drop('test1',axis=1,inplace=True)
data.drop('test2',axis=1,inplace=True)
def sigmoid(z):
    return 1/(1+ np.exp(-z))
def costreg(thetav,xv,yv,lr):
    thetav=np.matrix(thetav)
    xv=np.matrix(xv)
    yv=np.matrix(yv)
    first=np.multiply(-yv,np.log(sigmoid(xv * thetav.T)))
    second=np.multiply((1 -yv),np.log(1-sigmoid(xv * thetav.T)))
    reg = (lr / 2 * len(xv)) * np.sum(np.power(thetav[:,1:thetav.shape[1]], 2))
    return np.sum(first - second) / (len(xv)) + reg
def gradientreg(thetav,xv,yv,lr):
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



cols=data.shape[1]
X2=data.iloc[:,1:cols]
y2=data.iloc[:,0:1]
X2=np.matrix(X2.values)
y2=np.matrix(y2.values)
theta2=np.zeros(X2.shape[1])
lr=1
rcost=costreg(theta2,X2,y2,lr)
print('reg cost =',rcost)
result = opt.fmin_tnc(func=rcost, x0=theta2, fprime=gradientreg,

                      args=(X2, y2, lr))

print( 'result = ' , result )

print()


def predict(theta, X):

    probability = sigmoid(X * theta.T)

    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.matrix(result[0])

predictions = predict(theta_min, X2)

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]

accuracy = (sum(map(int, correct)) % len(correct))

print ('accuracy = {0}%'.format(accuracy))
        