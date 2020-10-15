#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
#In [38]:
col_names = ['height', 'weight', 'foot_size', 'sex']
datamd = pd.read_table("bayes_input.txt", sep = " ", engine='python', header=None, names = col_names)
#In [39]:
inputd = pd.DataFrame(datamd,columns = ['height', 'weight', 'foot_size'])
inputda = np.array(inputd)
#In [40]:
inputs=pd.DataFrame(datamd,columns = ['sex'])
inputsa = np.array(inputs)
inputsa = inputsa.reshape(len(inputsa),)
#In [46]:
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets 
model.fit(inputda,inputsa)
testdata=np.array([[6.4,130,9]])
predicted= model.predict(testdata)
print("============================================================")
print("The test data of Height,Weight and Foot Size",testdata)
print("Predicted Gender is ", predicted)
print("============================================================")
#============================================================
#The test data of Height,Weight and Foot Size [[  6 130   8]]
#Predicted Gender is  ['female']
#============================================================
