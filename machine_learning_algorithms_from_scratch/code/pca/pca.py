from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print("The CoVariance of data is",V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print("The Eigen Vectors are",vectors)
print("The Eigen Values are",values)
# project data
P = vectors.T.dot(C.T)
print("The Projected values are")
print(P.T)
