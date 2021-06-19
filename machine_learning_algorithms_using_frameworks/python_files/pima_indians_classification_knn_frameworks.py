# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = '../../datasets/pima-indians_classification_train.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
inputx = array[:,0:8]
outputy = array[:,8]
num_folds = 10
kfold = KFold(n_splits=10, random_state=None)
model = KNeighborsClassifier()
print("\nThe selected model is ",model)
results = cross_val_score(model, inputx, outputy, cv=kfold)
print("\nThe mean of the accuracy of cross validation results is",results.mean())
model.fit(inputx,outputy)
filename = '../../datasets/pima-indians_classification_test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
inputx = array[:,0:8]
print("\nThe test data is as follows\n",inputx)
results = model.predict(inputx)
print("\nThe predicted output for the data data is ", model.predict(inputx))
for val in results:
    if val == 0:
        print("diabetes not probable",end="   ")
    else:
        print("probability of getting diabetes",end="   ")
print()
