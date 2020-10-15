from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target
print(type(iris.target))
print(y)
#print(y)
dtree=DecisionTreeClassifier()
dtree.fit(df,y)
dot_data = StringIO()
print(dot_data)
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("dtree.png")
#Image(graph.create_png())
