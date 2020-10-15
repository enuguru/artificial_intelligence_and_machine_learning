'''
Code for Kfold crossvalidation

written by @srinadhu
'''

import smo  #My implementation of SMO
from sklearn import svm #inbuilt svm implementation
import numpy as np
import math


def Error(X_train,Y_train,alpha,bias,sigma):
	''' Error for the test data'''
	
	Y_t_predict = np.zeros(shape=(Y_train.shape[0],1))

	for i in range(X_train.shape[0]):
		if (smo.predict(X_train,Y_train,alpha,bias,X_train[i,:],sigma) >= 0 ):
			Y_t_predict[i]=1.0
		else:
			Y_t_predict[i]=-1.0

	train_error=0.0
	
	for i in range(Y_t_predict.shape[0]):
		if (Y_t_predict[i]!=Y_train[i]):
			train_error+=1.0

	return (1-(train_error/Y_train.shape[0]))*100.0

def Matrices(filename,k):
	'''returns the file input into matrices for both data and labels'''

	labels=[]	

	data=[]

	f=open(filename)  #opening for reading

	for line in f:
		temp=line.split(" ")
		try:
			if (float(temp[0])==2.0): #one class
				labels.append(float(-1.0)) #labels for data
			else:  #other class
				labels.append(float(1.0))
		except:
			continue
		temp=temp[1:] #all features.let's do a unit vector normalization.

		for i in range(len(temp)):
			try:
				temp[i]=float(temp[i])
			except:
				del(temp[i])

		data.append(temp)

	f.close()

	test_rows=int(len(data)/10)
	train_rows=len(data)-test_rows

	X_train=np.zeros(shape=(train_rows,len(data[i]))) #no of examples and no of features
	Y_train=np.zeros(shape=(train_rows,1)) #for labels

	X_test=np.zeros(shape=(test_rows,len(data[i]))) #no of examples and no of features
	Y_test=np.zeros(shape=(test_rows,1)) #for labels
	count=-1
	for i in range(len(data)): #for each row
		if ( (k-1)*test_rows <= i and i<k*test_rows): #the test examples
			X_test[i-((k-1)*test_rows),:]=data[i]
			Y_test[i-((k-1)*test_rows),:]=labels[i]
		else:
			count+=1
			X_train[count,:]=data[i]
			Y_train[count,:]=labels[i]

	return X_train,Y_train,X_test,Y_test


K=10 #kfold of 10
Train_error=[]
for k in range(1,K+1):
	X_train,Y_train,X_test,Y_test=Matrices("breast-cancer_scale",k)
	print "called SMO"
	alpha,bias= smo.SMO(X_train,Y_train,1.0,math.pow(10,-3),5,0.4)  #with best one on all training data.
	print "done"
	tr_err= Error(X_test,Y_test,alpha,bias,0.4)
	Train_error.append(tr_err)
	print tr_err
print Train_error



