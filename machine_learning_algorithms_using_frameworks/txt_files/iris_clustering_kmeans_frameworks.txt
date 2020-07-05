
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#import the dataset
filename = '../../datasets/iris_clustering_train.csv'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.read_csv(filename, names=names)

array = df.values
inputx = array[:,0:4]
inputx = df.iloc[:, [0,1,2,3]].values

kmeans3 = KMeans(n_clusters=3,verbose=1)
print(kmeans3)
y_kmeans3 = kmeans3.fit_predict(inputx)
print(y_kmeans3)
print(kmeans3.cluster_centers_)

plt.scatter(inputx[:,0],inputx[:,1],c=y_kmeans3,cmap='rainbow')    
plt.show()
