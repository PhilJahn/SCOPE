from DBHDALGO import DBHD
from sklearn.datasets import load_digits
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
from readFile import read_file



#X, Y = load_digits(return_X_y=True) #dataSet, labels
#X, Y = np.load('/Users/walid/PycharmProjects/streaming5/dbhd_clustering/mc_1000/mc_1000.npy'), None
X,  Y = read_file('optdigits',1)
#idx = np.random.permutation(len(X))
#X, Y = X[idx], Y[idx]
#print('len', len(X))
#n, rho, beta = (60, 1, 0.1) # minclusterSize, rho, beta parameter
from math import log2
n = int(log2(X.shape[0])) + 5

num_samples = 1000 # int(0.5 * len(X))
print(num_samples)
indices = np.random.choice(len(X), num_samples, replace=False)

X = X[indices]
Y = Y[indices]
from sklearn.preprocessing import  MinMaxScaler
X = MinMaxScaler().fit_transform(X)
dbhd = DBHD(n)
y_pred = dbhd.fit_predict(X)    #DBHD(X, n, rho, beta) #return Labels
#plt.scatter(X[:,0], X[:,1], c=y_pred.tolist())
#plt.xlim(-0.2,1.2)
#plt.ylim(-0.2,1.2)

plt.show()
print(f'AMI value {adjusted_mutual_info_score(Y, y_pred)}')



