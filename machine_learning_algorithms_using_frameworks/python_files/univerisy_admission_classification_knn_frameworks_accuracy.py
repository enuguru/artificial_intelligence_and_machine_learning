# KNN Classification

# step 1: import the required modules
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# step 2: read the data from the csv file
filename = '../../datasets/university_admission_classification_train.csv'
names = ['selection', 'gre', 'gpa', 'prestige']
df = read_csv(filename, names=names)

# step 3: use label encoder to convert label to numbers for the last column - prestige
# label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
df['prestige']= label_encoder.fit_transform(df['prestige'])
df['prestige'].unique()

# step 4: split in to input and output
array = df.values
inputx = array[:,1:4]
outputy = array[:,0]

# step 5: use cross validation
kfold = KFold(n_splits=10,random_state=None,shuffle=False)

# step 6: select KNN model
model = KNeighborsClassifier()
print("\nThe selected model is",model)

# step 7: train or build the model
model.fit(inputx,outputy)


# step 8: perform cross validation with accuracy as the performance metric
scoring = 'accuracy'
results = cross_val_score(model,inputx,outputy,cv = kfold, scoring = scoring)
print("\nThe Accuracy of the model is: %.3f (%.3f)" % (results.mean(), results.std()))




