# Cross Validation Classification Accuracy

# step 1: import the required modules
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# step 2: acquire the data from the csv file
filename = '../../datasets/pima-indians_classification_train.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)

# step 3: split in to input and output
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# step 4: split in to 10 folds
kfold = KFold(n_splits=10, random_state=None,shuffle=False)

# step 5: select the KNN model
model = KNeighborsClassifier()

# step 6: select accuracy as the model metric and perform cross validation 
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
