
# KNN Classification
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier


# step 1: reading the data and splitting it to input and output
filename = '../../datasets/bankloan_classification_train.csv'
headernames = ['age', 'loanamount', 'status']
df = read_csv(filename, names=headernames)
array = df.values
inputx = array[:,0:2]
outputy = array[:,2]


# step 2: selecting the KNN model 
thismodel = KNeighborsClassifier()#n_neighbors=3)
print("\nThe model selected is",thismodel)
print("\nThe parameters of the model are\n\n",thismodel.get_params())
#print(thismodel.set_params())


# step 3: training the model
thismodel.fit(inputx,outputy)


# step 4: testing and model prediction
filename = '../../datasets/bankloan_classification_test.csv'
names = ['age', 'loanamount']
newdataframe = read_csv(filename, names=names)
array = newdataframe.values
testinputz = array[0:4,0:2]
print("\n\nThe test inputs are\n\n",newdataframe)
res=thismodel.predict(testinputz)


# step 5: visualizing the test results
reslist=[]
for val in res:
    if val==0:
        reslist.append("WillNotPay")
    else:
        reslist.append("WillPay")
print("\nThe test results are\n\n",reslist)
#res=thismodel.predict(testinputz)
#print(thismodel.predict(testinputz))



