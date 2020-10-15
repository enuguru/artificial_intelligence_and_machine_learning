'''
Written by @srinadhu on Nov 19th.

reference: http://cs229.stanford.edu/materials/smo.pdf

'''


import numpy as np #for dealing with matrices.
import math #for exponential and power function
import random #for random number generation
import copy



def gaussian_kernel(x1,x2,sigma=0.5):
	'''returns the dot product in infinite dimensional space'''

	norm= np.linalg.norm(np.subtract(x1,x2)) #norm 

	res= math.exp(-(norm**2)/(2*(sigma**2))) #returning the final dot product.
	
	return res

def polynomial_kernel(x1,x2,degree=1):
	'''returns the dot in trnasformed polynomial space'''

	dot_prdt= np.dot (np.transpose(x1),x2) #give the dot product 

	return (dot_prdt+1)**degree  #returning the final dot product.


def predict(X,Y,alpha,b,x,sigma):
	'''predict the value for a new data point'''

	result=0.0

	for i in range(X.shape[0]):
		result+=(alpha[i]*Y[i]*polynomial_kernel(X[i,:] , x,sigma));

	result+=b

	return result

def SMO(X,Y,C=0.05,tol=math.pow(10,-3),max_passes=50,sigma=1):
	''' X has input data matrix. Y has the class labels. C is regularization parameter. tol is numerical tolerance. max_passes is max # of times to iterate wihtout changing alpha's

        Return Alpha and b.'''

	alpha=np.zeros(shape=(X.shape[0],1)); # each alpha[i] for every example.
	b=0.0
	
	passes=0

	E=np.zeros(shape=(X.shape[0],1)) #will be used in the loop
	alpha_old=copy.deepcopy(alpha) #deepcopy otherwise will do a shallow copy which will result in unwanted change of variables

	while(passes < max_passes):
		num_changed_alphas=0
		for i in range(X.shape[0]): #for every example
			E[i]=(predict(X,Y,alpha,b,X[i,:],sigma)-Y[i])
 		
			if ( (-Y[i]*E[i]>tol and -alpha[i]>-C) or (Y[i]*E[i]>tol and alpha[i]>0) ):
				j=i
				while(j==i):
					j=random.randrange(X.shape[0]) #get any other data point other than i
	
				E[j] = (predict(X,Y,alpha,b,X[j,:],sigma)-Y[j]) #for other data point

				alpha_old[i]=alpha[i]
				alpha_old[j]=alpha[j]
				
				#computing L and h values

				if (Y[i]!=Y[j]):
					L=max(0,alpha[j]-alpha[i])
					H=min(C,C+alpha[j]-alpha[i])
				else:
					L=max(0,alpha[i]+alpha[j]-C)
					H=min(C,alpha[i]+alpha[j])
		
				if (L==H):
					continue
				eta = 2*polynomial_kernel(X[i,:],X[j,:],sigma)
				eta=eta-polynomial_kernel(X[i,:],X[i,:],sigma)
				eta=eta-polynomial_kernel(X[j,:],X[j,:],sigma)
			
				if (eta >= 0):
					continue
			
				#clipping
				
				alpha[j]= alpha_old[j]-((Y[j]*(E[i]-E[j]))/eta)

				if (alpha[j] > H):
					alpha[j]=H
				elif (alpha[j]<L):
					alpha[j]=L
				else:
					pass  #do nothing
	
				if (abs(alpha[j]-alpha_old[j]) < tol):
					continue
			
				alpha[i] += (Y[i]*Y[j]*(alpha_old[j] - alpha[j])) #both alphas are updated


				ii = polynomial_kernel(X[i,:],X[i,:],sigma)
				ij = polynomial_kernel(X[i,:],X[j,:],sigma)
				jj = polynomial_kernel(X[j,:],X[j,:],sigma)			

				b1= b-E[i]- (Y[i]*ii*(alpha[i]-alpha_old[i]))- (Y[j]*ij*(alpha[j]-alpha_old[j]))
				b2= b-E[j]- (Y[i]*ij*(alpha[i]-alpha_old[i]))- (Y[j]*jj*(alpha[j]-alpha_old[j]))
				if (alpha[i] > 0 and alpha[i]<C):
					b=b1
				elif (alpha[j] > 0 and alpha[j] <C):
					b=b2
				else:
					b=(b1+b2)/2.0
			
				num_changed_alphas+=1
			#ended if
		#ended for
		if (num_changed_alphas == 0):
			passes+=1
		else:
			passes=0
	#end while

	return alpha,b   #returning the lagrange multipliers and bias.
					
