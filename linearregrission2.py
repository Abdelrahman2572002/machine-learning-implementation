import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data
path = 'C:\\abdelrahman\\exdata.txt'
data = pd.read_csv(path, header=None , names=['population','profit'])
path2 = 'C:\\abdelrahman\\exldata2.txt'
data2= pd.read_csv(path2, header=None , names=['size','bedrooms','price'])
print('data2= ')
print(data2.head(10))
print('data describe = ')
print(data2.describe())

#rescal data
data2= (data2-data2.mean()) / data2.std()
print('data after normalization = ')
print(data2.head(10))

data2.insert(0,'ones',1)

#separate data
cols2= data2.shape[1]
X2= data2.iloc[:,0:cols2-1]
Y2= data2.iloc[:,cols2-1:cols2]
#convert to matrix
X2=np.matrix(X2.values)
Y2=np.matrix(Y2.values)
theta2=np.matrix(np.matrix([0,0,0]))

alpha=0.1
iters=1000
def computecost(X,Y,theta):
    z=np.power(((X * theta.T)-Y),2)
    return np.sum(z) / (2* len(X))
def gradientdecent(X,Y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    paramters= int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    for i in range(iters):
        error= (X *theta.T) -Y
        
        for j in range(paramters):
            term=np.multiply(error,X[:,j])
            temp[0,j]= theta [0,j]- ((alpha / len(X)) * np.sum(term))
            
        theta=temp
        cost[i]= computecost(X, Y, theta)
        
    return theta,cost
g2,cost2= gradientdecent(X2, Y2, theta2, alpha, iters)
thiscost=computecost(X2, Y2, theta2)
print('g2 =',g2)
print('cost =',cost2[0:50])
print('computecost =',thiscost )

print('size shape', data2.iloc[:,1].shape)
print('price shape ', data2.price.shape)
##x =np.linspace(data2.size.min(), data2.size.max(),47)
f= g2[0,0]+(g2[0,1]*data2.iloc[:,1])
print('f =',f)
fig,ax= plt.subplots(figsize=(5,5))
ax.plot(data2.iloc[:,1],f,'r',label='predicition')
ax.scatter(data2.iloc[:,1],data2.price,label='training data')
ax.legend(loc=2)
ax.set_xlabel('size')
ax.set_ylabel('price')
ax.set_title('price vs size')
f= g2[0,0]+(g2[0,1]*data2.iloc[:,2])
print('f =',f)
fig,ax= plt.subplots(figsize=(5,5))
ax.plot(data2.iloc[:,2],f,'r',label='predicition')
ax.scatter(data2.iloc[:,2],data2.price,label='training data')
ax.legend(loc=2)
ax.set_xlabel('bedrooms')
ax.set_ylabel('price')
ax.set_title('price vs bedrooms')
#draw error graph
fig,ax= plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters),cost2,'r')
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
ax.set_title('error')










