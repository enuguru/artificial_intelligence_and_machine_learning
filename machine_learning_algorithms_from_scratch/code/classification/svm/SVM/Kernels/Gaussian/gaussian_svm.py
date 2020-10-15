'''
SVM for Linearly seperable data. Small demonstration

Written by @srinadhu on Nov 26th.

Data from http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex8/ex8.html

'''

import smo  #has the code for optimizer
import svm #code for basic SVM for general dataset
import matplotlib.pyplot as plt #for plotting the decision boundary.
import numpy as np
import math
import random

	

	

def main():
	''' All procedures done through this '''

	X_train,Y_train=svm.Matrices("data","no") #get the data into matrixes
	
	sigma=100

	alpha,bias= smo.SMO(X_train,Y_train,2.0,math.pow(10,-5),25,sigma)  #Calling the SMO procedure for getiing alpha and bias
	print alpha
	print bias
	print "done"
	#fig, ax = plt.subplots()
	#grid=plot_decision_boundary(X_train,Y_train,alpha,bias,sigma)	 #plot the decision boundary
	#plt.savefig("kernel.png")
	#print grid
	return 1

def decision_function(alpha,y,X,x_test,bias,sigma):
	'''returns which class it belongs to'''

	fun_value=smo.predict(X,y,alpha,bias,x_test,sigma)

	if (fun_value>=0.0):
		return 1.0
	else:
		return -1.0

def plot_decision_boundary(X,y,alpha,bias,sigma, resolution=300, colors=('b', 'k', 'r')):
	'''Plotting the decision boundary'''
	xrg = np.linspace(X[:,0].min(), X[:,0].max(), resolution)
	yrg = np.linspace(X[:,1].min(), X[:,1].max(), resolution)
	grid = [[decision_function(alpha,y,X,np.array([xr, yr]),bias,sigma) for yr in yrg] for xr in xrg]
	grid = np.array(grid).reshape(len(xrg), len(yrg))
        
	plt.contour(xrg, yrg, grid, (-1, 0, 1), linewidths=(1, 1, 1),linestyles=('--', '-', '--'), colors=colors)	
	plt.show()
	
	plt.scatter(np.multiply(xrg,np.ones(shape=(len(xrg),len(xrg)))),np.multiply(yrg,np.ones(shape=(len(xrg),len(xrg)))),c=grid, cmap=plt.cm.viridis,alpha=0.5)
	plt.show()

	print "done"
main()

