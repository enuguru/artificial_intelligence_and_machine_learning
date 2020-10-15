'''
Written by @srinadhu on Nov 19th.

'''

import smo  #has the code for optimizer
import matplotlib.pyplot as plt #for plotting the decision boundary.
import numpy as np
import math

def Error(X_train,Y_train,alpha,bias,X_test,Y_test,sigma):
	''' Error for the test data'''

	Y_predict =np.zeros(shape=(Y_test.shape[0],1)) #predicted by svm
	Y_t_predict = np.zeros(shape=(Y_train.shape[0],1))
	for i in range(X_test.shape[0]):
		if (smo.predict(X_train,Y_train,alpha,bias,X_test[i,:],sigma) >= 0 ):
			Y_predict[i]=1.0
		else:
			Y_predict[i]=-1.0

	for i in range(X_train.shape[0]):
		if (smo.predict(X_train,Y_train,alpha,bias,X_train[i,:],sigma) >= 0 ):
			Y_t_predict[i]=1.0
		else:
			Y_t_predict[i]=-1.0

	test_error=0.0
	train_error=0.0
	
	for i in range(Y_predict.shape[0]):
		if (Y_predict[i]!=Y_test[i]):
			test_error+=1.0

	
	for i in range(Y_t_predict.shape[0]):
		if (Y_t_predict[i]!=Y_train[i]):
			train_error+=1.0

	return (1-(train_error/Y_train.shape[0]))*100.0,(1-(test_error/Y_test.shape[0]))*100.0

def Matrices(filename,normalization="yes"):
	'''returns the file input into matrices for both data and labels'''

	labels=[]	

	data=[]

	f=open(filename)  #opening for reading

	for line in f:
		temp=line.split("\t")
		try:
			if (float(temp[0])==0.0):
				labels.append(float(-1.0)) #labels for data
			else:
				labels.append(float(1.0))
		except:
			continue
		temp=temp[1:] #all features.let's do a unit vector normalization.

		for i in range(len(temp)):
			temp[i]=float(temp[i])

		if (normalization == "yes"):
			norm=np.linalg.norm(temp) #norm of the input data
		
			for i in range(len(temp)):
				temp[i]= temp[i]/norm   #normalizing it to 1 and 0.		

		data.append(temp)

	f.close()

	X=np.zeros(shape=(len(data),len(data[i]))) #no of examples and no of features
	Y=np.zeros(shape=(len(data),1)) #for labels

	for i in range(X.shape[0]): #for each row
		X[i,:]=data[i]
		Y[i,:]=labels[i]

	return X,Y

def alphas(alpha):
	'''returns the number of ranges of them'''
	a=0
	for i in range(alpha.shape[0]):
		if (alpha[i] > 0.0):
			a+=1
		
	return a

def plot(Train_error,Test_error,support_vectors,Sigmas):
	'''returns the plots'''
	
	plt.plot(Sigmas,Train_error,color='r')
	plt.plot(Sigmas,Test_error,color='b')
	plt.xlabel("Degree")
	plt.ylabel("Train & Test Accuracy")
	plt.title("Accuracy vs Degree of Polynomial Kernel. \n(r-train\nb-test)\n")
	plt.savefig("./class_error.png", bbox_inches='tight')
	plt.clf()


	plt.plot(Sigmas,support_vectors,color='r')
	plt.xlabel("Degree")
	plt.ylabel("No of Support Vectors")
	plt.title("No of Support Vectors vs Degree.")
	plt.savefig("./support_vectors.png", bbox_inches='tight')
	plt.clf()

