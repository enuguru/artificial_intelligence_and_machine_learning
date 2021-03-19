
# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import ensemble
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(inputx, outputy)
#array = df.values
#inputx = array[1:20,0:4]
print(eclf1.predict(inputx))

np.array_equal(eclf1.named_estimators_.lr.predict(inputx),
               eclf1.named_estimators_['lr'].predict(inputx))

eclf2 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft')
eclf2 = eclf2.fit(inputx, outputy)
#print(eclf2.predict(inputx))

eclf3 = VotingClassifier(estimators=[
       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
       voting='soft', weights=[2,1,1],
       flatten_transform=True)
eclf3 = eclf3.fit(inputx, outputy)
#print(eclf3.predict(inputx))
#model = ensemble.AdaBoostClassifier()
#print(eclf2.fit(inputx,outputy))
#filename = '../../datasets/iris_classification_test.csv'
#names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#newdataframe = read_csv(filename, names=names)
#array = newdataframe.values
#z = array[:,0:4]
#print("\n",newdataframe,"\n")
#res=eclf2.predict(z)
#reslist=[]
#res=eclf2.predict(z)
#print(eclf2.predict(z),"\n")
#for val in res:
#    if val==0:
#        print("Iris_Setosa",end=" ")
#    elif val == 1:
#        print("Iris_Versicolor",end=" ")
#    else:
#        print("Iris_Viginica",end=" ")
#    print(end=" ")
#print("\n\n")
