
# python implementation of simple Linear Regression on salary data of software engineers

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('../../datasets/salary_regression_train.csv')
inputx = dataset.iloc[:, :-1].values
outputy = dataset.iloc[:, 1].values

input_train, input_test, output_train, output_test = train_test_split(inputx, outputy, test_size = 1/3, random_state = 0)
print(input_test)

# using simple Linear Regression model to train
model = LinearRegression()
model.fit(input_train, output_train)

# model predicting the Test set results
predicted_output = model.predict(input_test)
print(predicted_output)
years = float(input("Give number of years of experience  "))
testinput = [[years]]
predicted_output = model.predict(testinput)
print('The number of years of experience is ',testinput) 
print('The salary is ',predicted_output) 
yes = input("Can I proceed")


# Visualising the training results
plt.scatter(input_train, output_train, color = 'red')
plt.plot(input_train, model.predict(input_train), color = 'yellow')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the testing results
plt.scatter(input_test, output_test, color = 'red')
plt.plot(input_train, model.predict(input_train), color = 'yellow')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
