# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
filename = 'train.csv'
names = ['age', 'loanamount', 'status']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:2]
Y = array[:,2]
#num_folds = 2
#kfold = KFold(n_splits=2, random_state=7)
model = KNeighborsClassifier()
print(model.fit(X,Y))
filename = 'bankloantest.csv'
names = ['age', 'loanamount']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
z = array[0:4:,0:2]
#print(model.predict(X))
print(newdataframe)
res=model.predict(z)
reslist=[]
for val in res:
    if val==0:
        reslist.append("WillNotPay")
    else:
        reslist.append("WillPay")
print(reslist)
res=model.predict(z)
print(model.predict(z))
#results = cross_val_score(model, X, Y, cv=kfold)
#print(results.mean())
#print(results)
