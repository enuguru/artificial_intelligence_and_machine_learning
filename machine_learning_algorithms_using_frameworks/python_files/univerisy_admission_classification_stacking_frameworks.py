# KNN Classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]
model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

model.fit(inputx,outputy)
filename = '../../datasets/university_admission_classification_small_test.csv'
names = ['gre', 'gpa', 'prestige']
newdf = read_csv(filename, names=names)
# label_encoder object knows how to understand word labels
label_encoder = preprocessing.LabelEncoder()
newdf['prestige']= label_encoder.fit_transform(newdf['prestige'])
newdf['prestige'].unique()
array = newdf.values
z = array[:,0:3]
print("\n",newdf,"\n")
res=model.predict(z)
reslist=[]
res=model.predict(z)
print(model.predict(z),"\n")
for val in res:
    if val==0:
        print( "not selected",end=" ")
    elif val == 1:
        print( "selected",end=" ")
    print(end=" ")
print("\n\n")
