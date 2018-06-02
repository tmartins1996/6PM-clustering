
# -*- coding: utf-8 -*-
"""
Created on Dez 17 16:38:28 2017

@group  DM 2017 Semester 1, Group 2 

@author: Martins T.
@author: Mendes R.
@author: Santos R.


dataset - 2017/10/10

"""
print(__doc__)


import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os




#generating MinMax file
if not(os.path.isfile('6PM_data_transformation.xlsx')):
    exec(open('6PM_data_preparation.py').read())

dataset= pd.read_excel("6PM_data_transformation.xlsx")
finalData = dataset[['MntAcessoriesPercent','MntBagsPercent','MntClothingPercent','MntAthleticPercent','MntShoesPercent']]


data= linkage(finalData, 'ward')
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(
        data,
        truncate_mode = 'lastp',
        p=20,
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=12,
        show_contracted=True)
plt.show()


"""
Elbow Method
"""

cluster_range=range(1,10)
cluster_errors = []
for num_clusters in cluster_range: 
    clusters=KMeans(num_clusters)
    clusters.fit(finalData)
    cluster_errors.append(clusters.inertia_)
    
clusters_df = pd.DataFrame({"num_clusters": cluster_range,"cluster_errors": cluster_errors})
print(clusters_df[0:10])

plt.figure(figsize=(10,5))
plt.xlabel("Clusters")
plt.ylabel("within-Cluster Sum of Squares")
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker='o')