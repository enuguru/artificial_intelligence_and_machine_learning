# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
filename = '../../datasets/university_admission_classification_train.csv'
names = ['selection', 'gre', 'gpa', 'prestige']
df = read_csv(filename, names=names)
# label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
df['prestige']= label_encoder.fit_transform(df['prestige'])
df['prestige'].unique()
array = df.values
inputx = array[:,1:4]
outputy = array[:,0]
kfold = KFold(n_splits=10,random_state=None,shuffle=False)
model = KNeighborsClassifier()
model.fit(inputx,outputy)
scoring = 'accuracy'
results = cross_val_score(model,inputx,outputy,cv = kfold, scoring = scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
