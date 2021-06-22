# KNN Classification

# step 1: import the required modules
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# step 2: read the data from the csv file
filename = '../../datasets/pima-indians_classification_train.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv(filename, names=names)

# step 3: split in to input and output
array = df.values
inputx = array[:,0:8]
outputy = array[:,8]

# step 4: select the model
model = KNeighborsClassifier()
print("\nThe model selected is",model)

# step 5: train or build the model
model.fit(inputx,outputy)

# step 6: do testing or model prediction
filename = '../../datasets/pima-indians_classification_test.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
z = array[:,0:8]
print("\n",newdataframe,"\n")
res=model.predict(z)

# step 7: visualize the test results
for val in res:
    if val==0:
        print("diabetes not probable",end=" ")
    elif val == 1:
        print("diabetes is probable",end=" ")
    print(end=" ")
print("\n\n")
