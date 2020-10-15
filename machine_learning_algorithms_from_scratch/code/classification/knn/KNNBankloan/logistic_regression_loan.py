# Logistic Regression Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
filename = 'bankloan.csv'
names = ['age', 'loanamount', 'status']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:2]
Y = array[:,2]
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver="liblinear")
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
print(model.fit(X,Y))
X = array[0:1,0:2]
print(model.predict(X))
