# Cross Validation Classification Accuracy
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
filename = '../../datasets/bankloan_classification_train.csv'
names = ['age', 'loanamount', 'status']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:2]
Y = array[:,2]
kfold = KFold(n_splits=10, random_state=None,shuffle=False)
model = KNeighborsClassifier()
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
