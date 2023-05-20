import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data
path = 'C:\\abdelrahman\\exdata.txt'
data = pd.read_csv(path, header=None , names=['population','profit'])
print('data=\n', data.head(10))
print('**********')
print('data.describe=\n' , data.describe())
print('***********')
data.plot(kind='scatter' , x='population' , y= 'profit',figsize=(5,5))
#new column before data
data.insert(0,'ones',1)
print('new data=\n' , data.head(10))
print('**********')
#sperate x from y
cols = data.shape[1]
X= data.iloc[:,0:cols-1]
y= data.iloc[:,cols-1:cols]

print('x data = \n' , X.head(10))
print('y data =\n' , y.head(10))
print('**********')

#convert data to matrecies
X= np.matrix(X.values)
y= np.matrix(y.values)
theta= np.matrix(np.array([0,0]))
print('x data = \n' , X)
print('x shape =' , X.shape)
print('theta \n' , theta)
print('theta shape =',theta.shape)
print('y \n',y)
print('y shape =' , y.shape)
print('********')

#cost function
def computeCost(X ,y,theta):
    z = np.power(((X * theta.T) -y) ,2)
    return np.sum(z) / (2 * len(X))
print('computecost =',computeCost(X ,y,theta))
print('*********')

#GD 
def gradientdescent(X , y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parametrs = int(theta.ravel().shape[1])
    cost= np.zeros(iters)
    
    for i in range(iters):
        error = (X* theta.T) -y
        
        for j in range(parametrs):
            term= np.multiply(error, X[:,j])
            temp[0,j]=theta[0,j] - ((alpha / len(X))*np.sum(term))
            
        theta=temp
        cost[i]=computeCost(X,y,theta)
        
    return theta ,cost


#read data
path2='C:\\abdelrahman\\exldata2.txt'
data2= pd.read_csv(path2, header=None, names=['Size', 'Bedrooms','Price'])

#show data
print('data =')
print(data2.head(10))
print()
print('data.describe =')
print(data2.describe())
#rescaling data
data2= (data2 - data2.mean()) / data2.std()
print()
print('data after normalization =')
print(data2.head(10))
#add ones coloumns
data2.insert(0,'ones',1)
cols= data2.shape[1]
x2= data2.iloc[:,0:cols-1]
y2= data2.iloc[:,cols-1:cols]
print('*****************')
print('x2 data = ',x2.head(10))
print('y2 data =',y2.head(10))
print('**************')
