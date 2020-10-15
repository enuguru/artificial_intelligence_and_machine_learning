'''
SVM for Linearly seperable data. Small demonstration

Written by @srinadhu on Nov 26th.

'''

import smo  #has the code for optimizer
import svm #code for basic SVM for general dataset
import matplotlib.pyplot as plt #for plotting the decision boundary.
import numpy as np
import math
import random

	

	

def main():
	''' All procedures done through this '''

	X_train,Y_train=svm.Matrices("aritificial_data","no") #get the data into matrixes

	deg_of_kernel=1 #since linearly seperable

	alpha,bias= smo.SMO(X_train,Y_train,0.25,math.pow(10,-3),50,deg_of_kernel)  #Calling the SMO procedure for getiing alpha and bias

	plot_decision_boundary(X_train,Y_train,alpha,bias)	 #plot the decision boundary

	return 1

def plot_decision_boundary(X_train,Y_train,alpha,bias):
	'''save the decision boundary plots'''

	w=np.zeros(shape=(1,X_train.shape[1])) #no of features.Here 2 for simplicity
 
	for i in range(X_train.shape[0]): #for each example

		w=np.add(w,np.multiply(alpha[i]*Y_train[i],X_train[i,:])) #getting w's.

		if (Y_train[i]==1.0): #one class
			plt.plot(X_train[i,0],X_train[i,1],'ro')

		else: #other class
			plt.plot(X_train[i,0],X_train[i,1],'bo')
	
	

	xx = np.linspace(0,10.0,1000) #varying till maximum data points
	yy = (-w[0,0] / w[0,1])* xx - (bias) / w[0,1] #caluclating other co-ordnate
	plt.plot(xx,yy,'k-')
	
	xx = np.linspace(0,10.0,1000) #varying till maximum data points
	yy = (-w[0,0] / w[0,1])* xx - (bias-1) / w[0,1] #caluclating other co-ordnate
	plt.plot(xx,yy,'c-')
	

	xx = np.linspace(0,10.0,1000) #varying till maximum data points
	yy = (-w[0,0] / w[0,1])* xx - (bias+1) / w[0,1] #caluclating other co-ordnate
	plt.plot(xx,yy,'m-')
	
	plt.xlim(0.0,7)
	plt.ylim(0.0,7)

	plt.xlabel("X1 feature")
	plt.ylabel("X2 feature")
	plt.text(6,6,"r:class 1\nb:class -1\n")
	plt.savefig('plot.jpeg')

main()

