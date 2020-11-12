#!/usr/bin/env python
# coding: utf-8

# In[9]:


# KNN Classification using PIMA Diabetes dataset
import pandas as pd
import numpy as np

filename = pd.read_csv('C:\\Users\\Meera\\Desktop\\Wipro AI-ML Training\\Scripts\\Day 2\\pima-indians-diabetes.data.csv')
print('Size of the dataset before cleansing is',filename.shape)


# In[10]:


filename.head()


# In[14]:


print("Total observations with Glucose as zero: ",filename[filename.plas == 0].shape[0])
print("Total observations with BP as zero: ",filename[filename.pres == 0].shape[0])
print("Total observations with BMI as zero: ",filename[filename.mass == 0].shape[0])


# In[34]:


# Removing observations with zero values
filename_clean = filename[(filename.plas != 0) & (filename.pres != 0) & (filename.mass != 0)]
print('Size of the dataset after cleansing is',filename_clean.shape)


# In[33]:


# Splitting the dataset into X for independent and y for dependent/target variable
feature_names_X = ['preg','plas','pres','skin','test','mass','pedi','age']
feature_names_y = ['class']
X = filename_clean[feature_names_X]
y = filename_clean[feature_names_y]


# In[35]:


# Importing libraries for KNN and cross validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[39]:


# Training the model using KFold cross validation
model = KNeighborsClassifier()
kfold = KFold(n_splits=10, random_state=10) 
score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
print(score)


# In[43]:


# Training using KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X,y)


# In[48]:


# Test 1 - Predict using the classifier
new_test1 = pd.DataFrame([[7,137,90,41,0,32,0.391,39]])
y_pred = classifier.predict(new_test1)
print(y_pred)

# The output is 0 which means the patient with above conditions is not diabetic
# Validated with Row # 757 in the dataset


# In[49]:


# Test 2 - Predict using the classifier
new_test2 = pd.DataFrame([[0,177,60,29,478,34.6,1.072,21]])
y_pred = classifier.predict(new_test2)
print(y_pred)

# The output is 1 which means the patient with above conditions is diabetic
# Validated with Row # 221 in the dataset

