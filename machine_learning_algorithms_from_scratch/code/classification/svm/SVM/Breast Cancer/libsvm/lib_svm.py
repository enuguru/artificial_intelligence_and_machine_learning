'''
Written by @srinadhu on Nov 26th.

'''
from sklearn import svm
import matplotlib.pyplot as plt #for plotting the decision boundary.
import numpy as np
import math

def Error(X_train,Y_train,model):
	''' Error for the test data'''

	
	Y_t_predict = np.zeros(shape=(Y_train.shape[0],1))
	

	for i in range(X_train.shape[0]):
			Y_t_predict[i]=model.predict([X_train[i,:]])

	
	train_error=0.0


	
	for i in range(Y_t_predict.shape[0]):
		if (Y_t_predict[i]!=Y_train[i]):
			train_error+=1.0

	return (1-(train_error/Y_train.shape[0]))*100.0

def Matrices(filename):
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

def plot(Train_error,Sigmas):
	'''returns the plots'''
	#plt.xlim(0.0,2.0)
	plt.plot(Sigmas,Train_error,color='r')
	plt.xlabel("Degree of Polynomial kernel")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Degree of gaussian Kernel. \n(C=1.0)")
	plt.savefig("./gaussian_kernel_libsvm.png", bbox_inches='tight')
	plt.clf()

	



X_train,Y_train=Matrices("breast-cancer_scale")


Train_error=[]

Sigmas=[]
support_vectors=[]

sigma=0

while(sigma<2):
	sigma+=0.1
	Sigmas.append(sigma)
	print "called SMO"
	model=svm.SVC(C=2.0,gamma=sigma)
	model.fit(X_train,Y_train)  #with varying sigma
	tr_err= Error(X_train,Y_train,model)
	Train_error.append(tr_err)
	print tr_err
	print "one call done"
print Train_error
plot(Train_error,Sigmas,)
