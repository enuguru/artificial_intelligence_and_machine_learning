
# coding: utf-8

# In[35]:

import pandas as pd
import numpy as np
from sklearn import tree
from os import system

def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

train=pd.read_csv("Titanic_Train.csv")
clean_data(train)


# In[36]:



target = train["Survived"].values
features = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values
decision_tree_gini = tree.DecisionTreeClassifier(criterion="gini",random_state = 1)
decision_tree_gini_ = decision_tree_gini.fit(features, target)
print("Decision Tree Using Gini Score:", decision_tree_gini_.score(features, target)) 

decision_tree_Igain = tree.DecisionTreeClassifier(criterion="entropy",random_state = 1)
decision_tree_Igain_ = decision_tree_Igain.fit(features, target)
print("Decision Tree Using Information Gain Score:", decision_tree_Igain_.score(features, target)) 


# In[37]:


feature_names = ["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]
tree.export_graphviz(decision_tree_gini_, feature_names = feature_names, out_file="tree_gini.dot")
tree.export_graphviz(decision_tree_Igain_, feature_names = feature_names, out_file="tree_Igain.dot")


# In[39]:


system("dot -Tpng tree_gini.dot -o Titanic_Decision_Tree_gini.png")
system("dot -Tpng tree_Igain.dot -o Titanic_Decision_Tree_IGain.png")


# In[40]:


test=pd.read_csv("Titanic_Test.csv")
clean_data(test)


# In[42]:



#extracting the features from the test set

test_features = test[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch"]].values

#making prediction on test set

prediction_gini = decision_tree_gini.predict(test_features)
prediction_Igain = decision_tree_Igain.predict(test_features)

#creating DataFrame

PassengerId = np.array(test["PassengerId"]).astype(int)

solution_gini = pd.DataFrame(prediction_gini, PassengerId, columns = ["Survived"])
solution_Igain = pd.DataFrame(prediction_Igain, PassengerId, columns = ["Survived"])

solution_gini.to_csv("Titanic_Gini_Result.csv", index_label = ["PassengerId"])
solution_Igain.to_csv("Titanic_Igain_Result.csv", index_label = ["PassengerId"])
