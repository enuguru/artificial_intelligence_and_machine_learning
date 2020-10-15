from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from pandas import read_csv
filename = 'play_tennis.csv'
names = ['outlook','temp','humidity','windy','play']
iris=pd.read_csv(filename,names=names)
print(iris)
#iris=datasets.load_iris()
df=pd.DataFrame(iris, columns=names)
df.target_name='play'
#print(df)
y = df[df.columns[4]]
print(y)
#y=df.target
dtree=DecisionTreeClassifier()
dtree.fit(df,y)
#dot_data = StringIO()
export_graphviz(dtree)
#export_graphviz(dtree, out_file=dot_data,
#                filled=True, rounded=True,
#                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("dtree.png")
#Image(graph.create_png())
