

# KNN Classification

# important note -- the default value of n_estimtors will change from 20 in sklearn version 0.20
# to 100 in sklearn version 0.22, so I have coded n_estimators = 100 to avoid the warning

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
filename = '../../datasets/pima-indians_classification_train.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
inputx = array[:,0:8]
outputy = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=None)
model = ExtraTreesClassifier(n_estimators=100)
results = cross_val_score(model, inputx, outputy, cv=kfold)
print(results.mean())
model.fit(inputx,outputy)
filename = '../../datasets/pima-indians_classification_test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
inputx = array[:,0:8]
print(inputx)
results = model.predict(inputx)
print(model.predict(inputx))
for val in results:
    if val == 0:
        print("diabetes not probable",end="   ")
    else:
        print("probability of getting diabetes",end="   ")
print()
