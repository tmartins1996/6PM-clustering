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
import seaborn as sns
import somoclu
import os




#generating MinMax file
if not(os.path.isfile('6PM_data_MinMax.xlsx')):
    exec(open('6PM_data_preparation.py').read())
    
dataset= pd.read_excel("6PM_data_MinMax.xlsx")
product = dataset[['Income_Per_Family_Household','Seniority_Months','Age','Recency','R_Mnt_Frq','Educ_Levels','NumWebPurchases']]

df= np.float32(product.values)

n_rows, n_columns=9,9

som=somoclu.Somoclu(n_columns,n_rows,gridtype='rectangular',neighborhood='bubble',initialization='pca')

som.train(df, epochs=22)
map_state=som.get_surface_state()
BMUs=som.get_bmus(map_state)
som.cluster()


som.view_umatrix(colorbar=True,figsize=(5,5),bestmatches=False)
clusters=som.clusters
som.view_component_planes(colorbar=True,figsize=(5,5))

