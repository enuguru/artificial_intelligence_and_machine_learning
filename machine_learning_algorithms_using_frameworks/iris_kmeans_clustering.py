
#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import the dataset
df = pd.read_csv('iris.csv')
print(df.head(10))
xtrain = [:,0:5]
ztest = [10:15,0:4]
kmeansmodel = KMeans(n_clusters=2, random_state=0).fit(xtest)
predictkmeansmodel.predict(z)

