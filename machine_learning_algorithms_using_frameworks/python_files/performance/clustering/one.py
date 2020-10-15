
#Kavita Madival.
#Clusters of Iris dataset.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/home/hp/RNT_Task/Dataset.csv')
x = dataset.iloc[:, [0,1, 2, 3]].values # First 4 rows

from sklearn.cluster import KMeans

#3 Clusters for setosa,vericolour,virginica
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Scatter plot of cluster.
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'virginica')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()
