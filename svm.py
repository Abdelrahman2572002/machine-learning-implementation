import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context('notebook')

sns.set_style('white')


from scipy.io import loadmat

from sklearn import svm


pd.set_option('display.notebook_repr_html', False)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 150)

pd.set_option('display.max_seq_items', None)
def plotdata(x,y,S):
    pos=(y==1).ravel()
    neg= (y==0).ravel()
    plt.scatter(x[pos,0],x[pos,1],s=S,c='b',marker='+',linewidths=1)
    plt.scatter(x[neg,0],x[neg,1],s=S,c='r',marker='o',linewidths=1)
   
    
def plot_svc(svc,x,y,h=0.02,pad=0.25):
    x_min,x_max=x[:,0].min()-pad,x[:,0].max()+pad
    y_min,y_max=x[:,1].min()-pad,x[:,1].max()+pad
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)


    plotdata(x, y,6)

   #plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)

   # Support vectors indicated in plot by vertical lines

    sv = svc.support_vectors_

    plt.scatter(sv[:,0], sv[:,1], c='y', marker='|', s=100, linewidths=5)

    plt.xlim(x_min, x_max)

    plt.ylim(y_min, y_max)

    plt.xlabel('X1')

    plt.ylabel('X2')

    plt.show()

    print('Number of support vectors: ', svc.support_.size)


data1=loadmat('C:\\abdelrahman\\ex6data1.mat')
print(data1)
y1=data1['y']
x1=data1['X']
plotdata(x1,y1,50)
clf= svm.SVC(C=1.0, kernel='linear')
clf.fit(x1,y1.ravel())
plot_svc(clf,x1,y1)

data2=loadmat('C:\\abdelrahman\\ex6data2.mat')
print(data2.keys())
y2=data2['y']
x2=data2['X']
plotdata(x2,y2,8)
clf2=svm.SVC(C=50,kernel='rbf',gamma=6)
clf2.fit(x2,y2.ravel())
plot_svc(clf2,x2,y2)
data3=loadmat('C:\\abdelrahman\\ex6data3.mat')
print(data3.keys())
y3=data3['y']
x3=data3['X']
plotdata(x3,y3,30)
clf3=svm.SVC(C=1.0,kernel='poly',degree=3,gamma=10)
clf3.fit(x3,y3.ravel())
plot_svc(clf3,x3,y3)
spam_train=loadmat('C:\\abdelrahman\\spamTrain.mat')
spam_test=loadmat('C:\\abdelrahman\\spamTest.mat')
X = spam_train['X']

Xtest = spam_test['Xtest']

y = spam_train['y'].ravel()

ytest = spam_test['ytest'].ravel()

svc = svm.SVC()

svc.fit(X, y)

print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))




