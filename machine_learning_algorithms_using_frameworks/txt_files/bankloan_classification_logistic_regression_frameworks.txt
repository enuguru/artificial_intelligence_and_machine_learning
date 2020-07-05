# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
filename = '../../datasets/bankloan_classification_train.csv'
names = ['age', 'loanamount', 'status']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:2]
Y = array[:,2]
model = LogisticRegression(solver="liblinear")
print(model.fit(X,Y))
filename = '../../datasets/bankloan_classification_test.csv'
names = ['age', 'loanamount']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
z = array[0:4:,0:2]
print(newdataframe)
res=model.predict(z)
reslist=[]
for val in res:
    if val==0:
        reslist.append("WillNotPay")
    else:
        reslist.append("WillPay")
print(reslist)
print(model.predict(z))
