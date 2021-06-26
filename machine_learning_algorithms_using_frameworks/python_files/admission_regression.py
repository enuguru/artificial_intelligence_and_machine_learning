
#!/usr/bin/env python
# coding: utf-8

# Importing Libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LinearRegression
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Importing Data 
dataset = pd.read_csv('Admission_Predict.csv')
array = dataset.values
inputx = array[:,0:8]
outputy = array[:,8]

input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/3, random_state = 0)
#print(input_test)

# using simple Linear Regression model to train
model = LinearRegression()
model.fit(input_train, output_train)

# model predicting the Test set results
predicted_output = model.predict(input_test)
print(predicted_output)

num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, input_train, output_train, cv=kfold, scoring=scoring)
print(("MSE: %.3f (%.3f)") % (results.mean(), results.std()))

