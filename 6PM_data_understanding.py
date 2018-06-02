# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:38:28 2017

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
from datetime import datetime
from scipy.stats import norm
import os

"""
LOAD DATA 
"""

dataset= pd.read_excel("6PM.xlsx")

#Droping columns regarding groups elements
columns_group_elements = ['Group', 'Element1','Element2','Element3']
dataset= dataset.drop(columns_group_elements, axis=1)



"""
DATA UNDERSTANDING - Excel Output  
"""
############        Writing on excel -  data understanding

set_describe=dataset.describe()
set_describe=set_describe.append(dataset.kurt(),ignore_index=True)
set_describe=set_describe.append(dataset.skew(),ignore_index=True)
set_describe=set_describe.append(dataset.median(),ignore_index=True)
set_describe=set_describe.append(dataset.isnull().sum().drop(['Marital_Status', 'Education','Dt_Customer']),ignore_index=True)
set_describe['New_index']=['count','mean','std','min','25%','50%','75%','max','kurt','skew','median','nulls']
set_describe.set_index('New_index')


#Education count and education % concatened in one dataframe
education_counts=dataset['Education'].value_counts()
education_percent=round((dataset['Education'].value_counts()/len(dataset['Education'])*100),2)
education = pd.concat([education_counts, education_percent], axis=1, join_axes=[education_counts.index])
education.columns=['education_counts','education_percent'] #renaming

marital_counts=dataset['Marital_Status'].value_counts()
marital_percent=round((dataset['Marital_Status'].value_counts()/len(dataset['Marital_Status'])*100),2)
marital = pd.concat([marital_counts,marital_percent], axis=1, join_axes=[marital_counts.index])
marital.columns=['marital_counts','marital_percent']


#delete if exists
if os.path.isfile('6PM_data_understanding.xlsx'):
    os.remove('6PM_data_understanding.xlsx')
    
#write to excel, in diferent sheets from each dataframe
writer = pd.ExcelWriter('6PM_data_understanding.xlsx')
set_describe.to_excel(writer, 'Numeric_DataFrame')
education.to_excel(writer, 'Education')
marital.to_excel(writer, 'Marital_Status')
writer.save()



"""
DATA UNDERSTANDING - PLOTTING - NUMERICAL
"""

# Set Seaborn Default Palette
sns.set_palette(sns.cubehelix_palette(8))


# INCOME
plot_Income = sns.distplot(dataset['Income'].dropna(), color="purple", kde=False)
plt.title("Income Distribution")
plt.show(plot_Income)

plot_Income = sns.boxplot(dataset['Income'].dropna(), orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Income Boxplot")
plt.show(plot_Income)


plot_MntAcessories = sns.distplot(dataset['MntAcessories'], color="purple", kde=False)
plt.title("Spent Acessories")
plt.show(plot_MntAcessories)

plot_MntAcessories = sns.boxplot(dataset['MntAcessories'], orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Spent Acessories")
plt.show(plot_MntAcessories)

plot_MntBags = sns.distplot(dataset['MntBags'], color="purple", kde=False, bins=20)
plt.title("Spent Bags")
plt.show(plot_MntBags)

plot_MntBags = sns.boxplot(dataset['MntBags'], orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Spent Bags")
plt.show(plot_MntBags)


plot_MntClothing = sns.distplot(dataset['MntClothing'].dropna(), color="purple", kde=False, bins=20)
plt.title("Spent Clothing")
plt.show(plot_MntClothing)

plot_MntClothing = sns.boxplot(dataset['MntClothing'], orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Spent Clothing")
plt.show(plot_MntClothing)

plot_MntAthletic = sns.distplot(dataset['MntAthletic'].dropna(), color="purple", kde=False, bins=20)
plt.title("Spent Athletic")
plt.show(plot_MntAthletic)

plot_MntAthletic = sns.boxplot(dataset['MntAthletic'], orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Spent Athletic")
plt.show(plot_MntAthletic)


plot_MntShoes = sns.distplot(dataset['MntShoes'], color="purple", kde=False)
plt.title("Spent Shoes")
plt.show(plot_MntShoes)

plot_MntShoes = sns.boxplot(dataset['MntShoes'], orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Spent Shoes")
plt.show(plot_MntShoes)


plot_MntPremiumProds = sns.distplot(dataset['MntPremiumProds'], color="purple", kde=False, bins=20)
plt.title("Spent Premium")
plt.show(plot_MntPremiumProds)


plot_MntPremiumProds = sns.boxplot(dataset['MntPremiumProds'], orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Spent Premium")
plt.show(plot_MntPremiumProds)


"""
DATA UNDERSTANDING - PLOTTING - CATEGORICAL
"""

# Education
plot_Education = sns.countplot(dataset['Education'], order=['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'],
                               palette=sns.cubehelix_palette(8))
plt.title("Education Frequency")
plt.show(plot_Education)

# Marital Status
plot_Marital_Status = sns.countplot(dataset['Marital_Status'],
                                    order=['Single', 'Divorced', 'Widow', 'Together', 'Married'],
                                    palette=sns.cubehelix_palette(8))
plt.title("Marital Status Frequency")
plt.show(plot_Marital_Status)


"""
DATA UNDERSTANDING - PLOTTING - NOMINAL
"""

Year_Birth = sns.distplot(dataset['Year_Birth'], color="violet", kde=False)
plt.show(Year_Birth)

Kidhome = sns.countplot(dataset['Kidhome'], order=[0, 1, 2], palette=sns.cubehelix_palette(8))
plt.show(Kidhome)

Teenhome = sns.countplot(dataset['Teenhome'], order=[0, 1, 2], palette=sns.cubehelix_palette(8))
plt.show(Teenhome)

Recency = sns.distplot(dataset['Recency'], color="violet", kde=False)
plt.show(Recency)

NumDealsPurchases = sns.countplot(dataset['NumDealsPurchases'], palette=sns.cubehelix_palette(8))
plt.show(NumDealsPurchases)

NumWebPurchases = sns.countplot(dataset['NumWebPurchases'], palette=sns.cubehelix_palette(8))
plt.show(NumWebPurchases)

NumCatalogPurchases = sns.countplot(dataset['NumCatalogPurchases'], palette=sns.cubehelix_palette(8))
plt.show(NumCatalogPurchases)

NumStorePurchases = sns.countplot(dataset['NumStorePurchases'], palette=sns.cubehelix_palette(8))
plt.show(NumStorePurchases)

NumWebVisitsMonth = sns.countplot(dataset['NumWebVisitsMonth'], palette=sns.cubehelix_palette(8))
plt.show(NumWebVisitsMonth)

"""
DATA UNDERSTANDING - PLOTTING - BINARY
"""

zeros = (dataset['AcceptedCmp1'].value_counts()[0], dataset['AcceptedCmp2'].value_counts()[0],
         dataset['AcceptedCmp3'].value_counts()[0], dataset['AcceptedCmp4'].value_counts()[0],
         dataset['AcceptedCmp5'].value_counts()[0], dataset['Complain'].value_counts()[0])
uns = (dataset['AcceptedCmp1'].value_counts()[1], dataset['AcceptedCmp2'].value_counts()[1],
       dataset['AcceptedCmp3'].value_counts()[1], dataset['AcceptedCmp4'].value_counts()[1],
       dataset['AcceptedCmp5'].value_counts()[1], dataset['Complain'].value_counts()[1],)
ind = np.arange(6)  # the x locations for the groups

p1 = plt.bar(ind, zeros, 0.5, color='xkcd:mauve')
p2 = plt.bar(ind, uns, 0.5,
             bottom=zeros, color='xkcd:lavender')

plt.ylabel('Count')
plt.title('Count by True or False ')
plt.xticks(ind, ('AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                 'AcceptedCmp4', 'AcceptedCmp5', 'Complain'), rotation=20)
plt.legend(('False', 'True'), loc="center right")
plt.show()


"""
DATA UNDERSTANDING - Integrity
"""
############        Integrity data

# dataset['Integrity_Num_Deals']=dataset['NumDealsPurchases']-dataset['Purchases_Cmp']
# dataset['Integrity_Num_Deals2']=dataset['Purchases_Cmp']-dataset['NumDealsPurchases']
# dataset['Integrity_Num_Discounts']=np.where(dataset['Purchases_Cmp']>dataset['NumDealsPurchases'],0,1)
# print(dataset['Integrity_Num_Discounts'].value_counts())


#dataset['Web_Visits-Deals']=dataset['NumWebVisitsMonth']-dataset['NumWebPurchases']
dataset['Web_Visits-Deals']=np.where((dataset['NumWebVisitsMonth']==0),(dataset['NumWebVisitsMonth']-dataset['NumWebPurchases']),0)

# Correlation?

plt.scatter((dataset.NumWebVisitsMonth * 12), dataset.NumWebPurchases, color='grey')
plt.show()
