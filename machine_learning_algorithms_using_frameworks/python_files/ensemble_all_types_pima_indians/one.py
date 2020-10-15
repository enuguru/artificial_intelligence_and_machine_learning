import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('iris_csv.csv', names=headernames)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print(x[0:5])
print(y[0:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 40)

model = RandomForestClassifier(n_estimators = 50)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

result = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ")
print(result)

result1 = classification_report(y_test, y_pred)
print("Classification report: ")
print(result1)

result2 = accuracy_score(y_test, y_pred)
print("Accuracy Score: ")
print(result2)
