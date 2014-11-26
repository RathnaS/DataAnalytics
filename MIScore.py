__author__ = 'Rathna'
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

from sklearn.metrics import mutual_info_score
def calc_MI(x,y):
    c_xy = np.histogram2d(x,y)[0]
    mi = mutual_info_score(None,None,contingency=c_xy)
    return mi

n = iris.data.shape[1]
matMI_c1 = np.zeros((n,n))
matMI_c2 = np.zeros((n,n))
matMI_c3 = np.zeros((n,n))

for i in np.arange(n):
    for j in np.arange(i+1,n):
        matMI_c1[i,j] = calc_MI(iris.data[1:50,i],iris.data[1:50,j])
        matMI_c2[i,j] = calc_MI(iris.data[51:100,i],iris.data[51:100,j])
        matMI_c3[i,j] = calc_MI(iris.data[101:150,i],iris.data[101:150,j])


print(matMI_c1)
print(matMI_c2)
print(matMI_c3)

