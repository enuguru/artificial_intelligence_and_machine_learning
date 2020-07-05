# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
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
model = LogisticRegression(solver='liblinear')
model.fit(inputx,outputy)
filename = '../../datasets/iris_classification_test.csv'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
z = array[:,0:4]
print("\n",newdataframe,"\n")
res=model.predict(z)
reslist=[]
res=model.predict(z)
print(model.predict(z),"\n")
for val in res:
    if val==0:
        print("Iris_Setosa",end=" ")
    elif val == 1:
        print("Iris_Versicolor",end=" ")
    else:
        print("Iris_Viginica",end=" ")
    print(end=" ")
print("\n\n")
