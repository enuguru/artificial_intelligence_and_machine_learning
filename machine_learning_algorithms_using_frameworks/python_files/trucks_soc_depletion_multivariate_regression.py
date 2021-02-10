
# python implementation of simple Linear Regression on salary data of software engineers

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../../datasets/truck_multi_variate_regression.csv')
inputx = dataset.iloc[:, :-1].values
outputy = dataset.iloc[:, 1].values

input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/3, random_state = 0)

model = LinearRegression()
model.fit(input_train, output_train)

predicted_output = model.predict(input_test)
plt.scatter(input_train, output_train, color = 'red')
plt.plot(input_train, model.predict(input_train), color = 'green')
plt.title('Multi Variate Regression')
plt.xlabel('predicted_state_of_charge')
plt.ylabel('Actual_state_of_charge')
plt.show()
