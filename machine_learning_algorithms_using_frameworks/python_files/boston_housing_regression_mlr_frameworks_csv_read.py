
# python implementation of simple Linear Regression on salary data of software engineers

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Importing Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
   
# Importing Data 
dataset = pd.read_csv('../../datasets/housing_regression_train.csv')
array = dataset.values
inputx = array[:,0:13]
outputy = array[:,13]

input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/3, random_state = 0)
#print(input_test)

# using simple Linear Regression model to train
model = LinearRegression()
model.fit(input_train, output_train)

# model predicting the Test set results
predicted_output = model.predict(input_test)
print(predicted_output)
