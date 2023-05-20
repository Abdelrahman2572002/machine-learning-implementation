import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data
path = 'C:\\abdelrahman\\exdata.txt'
data = pd.read_csv(path, header=None , names=['population','profit'])
#show data 
print('data = \n',data.head(10))
print('data describe = \n',data.describe())
#draw data
data.plot(kind='scatter',x='population',y='profit',figsize=(5,5))
data.insert(0,'Ones',1)
print('data = \n',data.head(10))
#separate data
cols= data.shape[1]
X=data.iloc[:,0:cols-1]
Y= data.iloc[:,cols-1:cols]
print('x data= \n',X.head(10))
print('y data= \n',Y.head(10))
#convert to matrix
X=np.matrix(X.values)
Y=np.matrix(Y.values)
theta= np.matrix(np.array([0,0]))

#cost function
def computecost(X,Y,theta):
    z=np.power(((X * theta.T)-Y),2)
    return np.sum(z) / (2* len(X))

print('compute cost function= \n', computecost(X, Y, theta))

#GD
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
    
#intilaize
alpha=0.01
iters=1000
g,cost= gradientdecent(X,Y,theta,alpha,iters)
print('g=',g)
print('cost = ', cost[0:50])
print('compute cost = ', computecost(X, Y, g))
#fit line
x=np.linspace(data.population.min(),data.population.max(),100)
print('x=',x)
print('g=',g)
f=g[0,0]+(g[0,1] * x)
print('f=',f)

#darw line
fig,ax= plt.subplots(figsize=(5,5))
ax.plot(x,f,'r',label='predicition')
ax.scatter(data.population,data.profit,label='training data')
ax.legend(loc=2)
ax.set_xlabel('population')
ax.set_ylabel('profit')
ax.set_title('profit vs population')


#draw error graph
fig,ax= plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
ax.set_title('error')





