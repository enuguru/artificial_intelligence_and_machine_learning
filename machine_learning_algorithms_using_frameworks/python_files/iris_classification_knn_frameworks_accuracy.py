# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
filename = '../../datasets/iris_classification_train.csv'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','flower_name']
df = read_csv(filename, names=names)
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
df['flower_name']= label_encoder.fit_transform(df['flower_name'])
df['flower_name'].unique()
array = df.values
inputx = array[:,0:4]
outputy = array[:,4]
kfold = KFold(n_splits=10,random_state=None,shuffle=False)
model = KNeighborsClassifier()
print(model.fit(inputx,outputy))
scoring = 'accuracy'
results = cross_val_score(model,inputx, outputy, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
