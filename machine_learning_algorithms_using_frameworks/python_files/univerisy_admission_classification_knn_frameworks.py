# KNN Classification

# step 1: import the required modules
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# step 2: read the data from the csv file 
filename = '../../datasets/university_admission_classification_train.csv'
names = ['selection', 'gre', 'gpa', 'prestige']
df = read_csv(filename, names=names)

# step 3: use label encoder to encode the "prestige" categorical value to a number
# this label encoding is during trainig
# label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
df['prestige']= label_encoder.fit_transform(df['prestige'])
df['prestige'].unique()

# step 4: split in to input and output
array = df.values
inputx = array[:,1:4]
outputy = array[:,0]

# step 5: select the model
model = KNeighborsClassifier()

# step 6: train or build the model
model.fit(inputx,outputy)
print("\nThe selected model is ", model)
filename = '../../datasets/university_admission_classification_small_test.csv'
names = ['gre', 'gpa', 'prestige']
newdf = read_csv(filename, names=names)

# step 7: use label encoder to encode the "prestige" categorical value to a number
# this lable encoding is during testing
# label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
newdf['prestige']= label_encoder.fit_transform(newdf['prestige'])
newdf['prestige'].unique()

# step 7: split in to input and output
array = newdf.values
z = array[:,0:3]
print("\nThe input test data is \n\n",newdf,"\n")

# step 8: testing and doing prediction using the model
res=model.predict(z)
reslist=[]
#res=model.predict(z)
print("The output / results of the test data are", model.predict(z),"\n")
for val in res:
    if val==0:
        print( "not selected",end=" ")
    elif val == 1:
        print( "selected",end=" ")
    print(end=" ")
print("\n")


