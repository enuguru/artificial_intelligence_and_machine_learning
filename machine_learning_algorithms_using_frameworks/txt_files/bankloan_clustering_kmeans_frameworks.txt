
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#import the dataset
filename = '../../datasets/bankloan_clustering_train.csv'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = pd.read_csv(filename, names=names)

array = df.values
inputx = array[:,0:2]
inputx = df.iloc[:, [0,1]].values

kmeans2 = KMeans(n_clusters=2)
print(kmeans2)
y_kmeans2 = kmeans2.fit_predict(inputx)
print(y_kmeans2)
print(kmeans2.cluster_centers_)

plt.scatter(inputx[:,0],inputx[:,1],c=y_kmeans2,cmap='rainbow')    
plt.show()
