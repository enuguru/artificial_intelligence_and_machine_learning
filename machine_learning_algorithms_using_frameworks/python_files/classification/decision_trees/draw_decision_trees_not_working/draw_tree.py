from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
filename = 'play_tennis.data'
names = ['outlook','temp','humidity','windy','play']
#play_tennis=datasets.load("play_tennis.data")
iris=pd.read_csv(filename,names=names)#dplay_tennisatasets.load_iris()
#df=pd.DataFrame(play_tennis.data, columns=iris.feature_names)
df=pd.DataFrame(iris, columns=names)
df.target_name='play'
y=df[df.columns[4]]#iris.target
dtree=DecisionTreeClassifier()
dtree.fit(df,y)
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("dtree.png")
#Image(graph.create_png())
