# KNN Classification

# step 1: import the required modules
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# step 2: read the data from the csv file and split it to input and output
filename = '../../datasets/iris_classification_train.csv'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','flower_name']
df = read_csv(filename, names=names)

# step 3: do pre-processing of data - in this case do label encoding on last column
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
df['flower_name']= label_encoder.fit_transform(df['flower_name'])
df['flower_name'].unique()

# step 4: split it to input and output
array = df.values
inputx = array[:,0:4]
outputy = array[:,4]

# step 5: split it in to many folds for doing cross validation
kfold = KFold(n_splits=10,random_state=None,shuffle=False)

# step 6: select the model
model = KNeighborsClassifier()

# step 7: do training of the model or build the model
print("\nThe model selected is",model.fit(inputx,outputy))

# step 8: calculate accuracy of the training process while doing cross validation
scoring = 'accuracy'
results = cross_val_score(model,inputx, outputy, cv=kfold, scoring=scoring)
print("\nThe model Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
