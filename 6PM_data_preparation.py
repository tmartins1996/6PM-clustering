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
from sklearn.preprocessing import MinMaxScaler
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os

"""
LOAD DATA 
"""

dataset= pd.read_excel("6PM.xlsx")

#Droping columns regarding groups elements
columns_group_elements = ['Group', 'Element1','Element2','Element3']
dataset= dataset.drop(columns_group_elements, axis=1)




"""
DATA PREPARATION 
"""

date_format = "%Y/%m/%d"




dataset['MntClothing'] = dataset['MntClothing'].fillna(dataset['MntClothing'].median())
dataset['MntAthletic'] = dataset['MntAthletic'].fillna(dataset['MntAthletic'].median())


#New column with value 2 for phd and master, else - if equal graduation 1, else 0
dataset['Educ_Levels']=np.where((dataset['Education']=='Master') | 
     (dataset['Education']=='PhD'),2,np.where((dataset['Education']=='Graduation'),1,0))


##Education count and education % concatened in one dataframe
#education_counts=dataset['Education'].value_counts()
#education_percent=round((dataset['Education'].value_counts()/len(dataset['Education'])*100),2)
#education = pd.concat([education_counts, education_percent], axis=1, join_axes=[education_counts.index])
#education.columns=['education_counts','education_percent'] #renaming
#
#marital_counts=dataset['Marital_Status'].value_counts()
#marital_percent=round((dataset['Marital_Status'].value_counts()/len(dataset['Marital_Status'])*100),2)
#marital = pd.concat([marital_counts,marital_percent], axis=1, join_axes=[marital_counts.index])
#marital.columns=['marital_counts','marital_percent']


dataset['Marital_Levels']=np.where((dataset['Marital_Status']=='Together') | 
     (dataset['Marital_Status']=='Married'),'Couple',dataset['Marital_Status'])

dataset['Marital_Levels_Binary']=np.where((dataset['Marital_Status']=='Together') | 
     (dataset['Marital_Status']=='Married'),1,0)


dataset['Purchases_Cmp_Binary']=np.where((dataset['AcceptedCmp1']==1) | 
     (dataset['AcceptedCmp2']==1) | 
     (dataset['AcceptedCmp3']==1)| 
     (dataset['AcceptedCmp4']==1)| 
     (dataset['AcceptedCmp5']==1),1,0)


#
dataset['Purchases_Cmp']=(dataset['AcceptedCmp1']
     +dataset['AcceptedCmp2'] +
     dataset['AcceptedCmp3'] + 
     dataset['AcceptedCmp4'] +
     dataset['AcceptedCmp5'])


dataset['Mnt_Total']=(dataset['MntAcessories']
     +dataset['MntBags'] +
     dataset['MntShoes'] + 
     dataset['MntClothing'] +
     dataset['MntAthletic'])


dataset['Total_Num_Purchases']=(dataset['NumWebPurchases']
     +dataset['NumCatalogPurchases'] +
     dataset['NumStorePurchases'])


dataset['Mnt_Regular']=(dataset['Mnt_Total']-dataset['MntPremiumProds'])



dataset['Family_Dependents']=dataset['Kidhome']+dataset['Teenhome']
Family_Dependentes=sns.countplot(dataset['Family_Dependents'], order=[0,1,2,3,4], palette=sns.cubehelix_palette(8))
plt.show(Family_Dependentes)



dataset['Date_Freezed']= datetime.strptime('2017/10/10', date_format)
dataset['Seniority']=(dataset['Date_Freezed']-dataset['Dt_Customer'])
dataset['Seniority_Years']=round(( (dataset['Seniority']/356)/ np.timedelta64(1, 'D')).astype(float),2)
dataset['Seniority_Months']=((dataset['Seniority']/30)/ np.timedelta64(1, 'D')).astype(int)



dataset['Age']=2017-dataset['Year_Birth']




dataset_Total = dataset[(dataset['Total_Num_Purchases']>0)]
predict_Total=dataset[(dataset['Total_Num_Purchases']<1)]

#criar arrays
trainingset_num=dataset_Total['Total_Num_Purchases'].values.reshape(6954, 1)
trainingset_total=dataset_Total['Mnt_Total'].values.reshape(6954, 1)

#criar arrays
testset_total=predict_Total['Mnt_Total'].values.reshape(46, 1)
testset_num=predict_Total['Total_Num_Purchases'].values.reshape(46, 1)


dataset['Total_Num_Purchases']=np.where((dataset['Total_Num_Purchases']==0),np.nan,dataset['Total_Num_Purchases'])



#########    Total_Num_Purchases    #########


# Create linear regression object
regr_Total_Num_Purchases= linear_model.LinearRegression()

# Train the model using the training sets
regr_Total_Num_Purchases.fit(trainingset_total, trainingset_num)

# Make predictions using the testing set

testset_num= regr_Total_Num_Purchases.predict(testset_total)

dataset['Total_Num_Purchases'][pd.isnull(dataset['Total_Num_Purchases'])]=np.round(regr_Total_Num_Purchases.predict(testset_total))

# The coefficients
print('Coefficients: \n', regr_Total_Num_Purchases.coef_)

#???
print('Intercept: \n', regr_Total_Num_Purchases.intercept_)


# Plot outputs
plt.scatter(trainingset_total, trainingset_num,  color='grey')
plt.plot(testset_total, testset_num, color='red', linewidth=2)
plt.show()



#dataset['Total_Num_Purchases']=np.where((dataset['Total_Num_Purchases']==0),1,dataset['Total_Num_Purchases'])
#dataset['R_Mnt_Frq']=(dataset['Mnt_Total']/np.where((dataset['Total_Num_Purchases']>0),dataset['Total_Num_Purchases'],1))
dataset['R_Mnt_Frq']=dataset['Mnt_Total']/dataset['Total_Num_Purchases']
dataset['Family_Dependents_Levels']=np.where((dataset['Family_Dependents']==0),0,np.where((dataset['Family_Dependents']==1),1,2))


    


#####   Regression     ######



trainingset=dataset[['Income','Mnt_Total']].dropna()
dataset.isnull().sum()
trainingset.count()

#criar arrays
trainingset_income=trainingset['Income'].values.reshape(6936, 1)
trainingset_total=trainingset['Mnt_Total'].values.reshape(6936, 1)




predict_income=dataset[['Income','Mnt_Total']][pd.isnull(dataset['Income'])]
dataset.isnull().sum()
predict_income.count()




#criar arrays
predict_income_bytotal=predict_income['Mnt_Total'].values.reshape(64, 1)



#########    INCOME    #########


# Create linear regression object
regr_income = linear_model.LinearRegression()

# Train the model using the training sets
regr_income.fit(trainingset_total, trainingset_income)

# Make predictions using the testing set

new_income= regr_income.predict(predict_income_bytotal)

dataset['Income'][pd.isnull(dataset['Income'])]=regr_income.predict(predict_income['Mnt_Total'].values.reshape(64, 1))

# The coefficients
print('Coefficients: \n', regr_income.coef_)

#???
print('Intercept: \n', regr_income.intercept_)


# Plot outputs
plt.scatter(trainingset_total, trainingset_income,  color='grey')
plt.plot(predict_income_bytotal, new_income, color='red', linewidth=2)
plt.show()

#append
predict_income['Income']=pd.DataFrame(new_income,predict_income.index ) 


plot_Income = sns.distplot(dataset['Income'].dropna(), color="purple", kde=False)
plt.title("Income Distribution")
plt.show(plot_Income)

plot_Income = sns.boxplot(dataset['Income'].dropna(), orient='h', saturation=0.5, whis=1.5, color="violet")
plt.title("Income Boxplot")
plt.show(plot_Income)






dataset['Income_Per_Income_Holders']=dataset['Income']/(dataset['Marital_Levels_Binary']+1)
dataset['Family_Household']=dataset['Kidhome']+dataset['Teenhome']+(dataset['Marital_Levels_Binary']+1)
dataset['Income_Per_Family_Household']=dataset['Income']/dataset['Family_Household']

dataset['MntAcessoriesPercent']=(dataset['MntAcessories']/dataset['Mnt_Total'])*100
dataset['MntBagsPercent']=(dataset['MntBags']/dataset['Mnt_Total'])*100
dataset['MntClothingPercent']=(dataset['MntClothing']/dataset['Mnt_Total'])*100
dataset['MntAthleticPercent']=(dataset['MntAthletic']/dataset['Mnt_Total'])*100
dataset['MntShoesPercent']=(dataset['MntShoes']/dataset['Mnt_Total'])*100

dataset['MntAcessoriesPercent']=np.round(dataset['MntAcessoriesPercent'],2)
dataset['MntBagsPercent']=np.round(dataset['MntBagsPercent'],2)
dataset['MntClothingPercent']=np.round(dataset['MntClothingPercent'],2)
dataset['MntAthleticPercent']=np.round(dataset['MntAthleticPercent'],2)
dataset['MntShoesPercent']=np.round(dataset['MntShoesPercent'],2)





dataset_norm=dataset[['Year_Birth','Income','Kidhome','Teenhome','Recency','MntAcessories','MntBags','MntClothing',
             'MntAthletic','MntShoes','MntPremiumProds','NumDealsPurchases','NumWebPurchases',
             'NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4',
             'AcceptedCmp5','Complain','Educ_Levels','Marital_Levels_Binary','Purchases_Cmp_Binary','Purchases_Cmp','Mnt_Total',
             'Total_Num_Purchases','Mnt_Regular','Family_Dependents','Seniority_Years','Seniority_Months','Age','R_Mnt_Frq',
             'Family_Dependents_Levels','Income_Per_Income_Holders','Family_Household','Income_Per_Family_Household']]


#Min Max foreach varible to keep as a dataframe instead of array

dataset_norm['Year_Birth'] = (dataset_norm['Year_Birth']-min(dataset_norm['Year_Birth']))/(max(dataset_norm['Year_Birth'])-min(dataset_norm['Year_Birth']))
dataset_norm['Income'] = (dataset_norm['Income']-min(dataset_norm['Income']))/(max(dataset_norm['Income'])-min(dataset_norm['Income']))
dataset_norm['Kidhome'] = (dataset_norm['Kidhome']-min(dataset_norm['Kidhome']))/(max(dataset_norm['Kidhome'])-min(dataset_norm['Kidhome']))
dataset_norm['Teenhome'] = (dataset_norm['Teenhome']-min(dataset_norm['Teenhome']))/(max(dataset_norm['Teenhome'])-min(dataset_norm['Teenhome']))
dataset_norm['Recency'] = (dataset_norm['Recency']-min(dataset_norm['Recency']))/(max(dataset_norm['Recency'])-min(dataset_norm['Recency']))
dataset_norm['MntAcessories'] = (dataset_norm['MntAcessories']-min(dataset_norm['MntAcessories']))/(max(dataset_norm['MntAcessories'])-min(dataset_norm['MntAcessories']))
dataset_norm['MntBags'] = (dataset_norm['MntBags']-min(dataset_norm['MntBags']))/(max(dataset_norm['MntBags'])-min(dataset_norm['MntBags']))
dataset_norm['MntClothing'] = (dataset_norm['MntClothing']-min(dataset_norm['MntClothing']))/(max(dataset_norm['MntClothing'])-min(dataset_norm['MntClothing']))
dataset_norm['MntAthletic'] = (dataset_norm['MntAthletic']-min(dataset_norm['MntAthletic']))/(max(dataset_norm['MntAthletic'])-min(dataset_norm['MntAthletic']))
dataset_norm['MntShoes'] = (dataset_norm['MntShoes']-min(dataset_norm['MntShoes']))/(max(dataset_norm['MntShoes'])-min(dataset_norm['MntShoes']))
dataset_norm['MntPremiumProds'] = (dataset_norm['MntPremiumProds']-min(dataset_norm['MntPremiumProds']))/(max(dataset_norm['MntPremiumProds'])-min(dataset_norm['MntPremiumProds']))
dataset_norm['NumDealsPurchases'] = (dataset_norm['NumDealsPurchases']-min(dataset_norm['NumDealsPurchases']))/(max(dataset_norm['NumDealsPurchases'])-min(dataset_norm['NumDealsPurchases']))
dataset_norm['NumWebPurchases'] = (dataset_norm['NumWebPurchases']-min(dataset_norm['NumWebPurchases']))/(max(dataset_norm['NumWebPurchases'])-min(dataset_norm['NumWebPurchases']))
dataset_norm['NumCatalogPurchases'] = (dataset_norm['NumCatalogPurchases']-min(dataset_norm['NumCatalogPurchases']))/(max(dataset_norm['NumCatalogPurchases'])-min(dataset_norm['NumCatalogPurchases']))
dataset_norm['NumStorePurchases'] = (dataset_norm['NumStorePurchases']-min(dataset_norm['NumStorePurchases']))/(max(dataset_norm['NumStorePurchases'])-min(dataset_norm['NumStorePurchases']))
dataset_norm['NumWebVisitsMonth'] = (dataset_norm['NumWebVisitsMonth']-min(dataset_norm['NumWebVisitsMonth']))/(max(dataset_norm['NumWebVisitsMonth'])-min(dataset_norm['NumWebVisitsMonth']))
dataset_norm['AcceptedCmp1'] = (dataset_norm['AcceptedCmp1']-min(dataset_norm['AcceptedCmp1']))/(max(dataset_norm['AcceptedCmp1'])-min(dataset_norm['AcceptedCmp1']))
dataset_norm['AcceptedCmp2'] = (dataset_norm['AcceptedCmp2']-min(dataset_norm['AcceptedCmp2']))/(max(dataset_norm['AcceptedCmp2'])-min(dataset_norm['AcceptedCmp2']))
dataset_norm['AcceptedCmp3'] = (dataset_norm['AcceptedCmp3']-min(dataset_norm['AcceptedCmp3']))/(max(dataset_norm['AcceptedCmp3'])-min(dataset_norm['AcceptedCmp3']))
dataset_norm['AcceptedCmp4'] = (dataset_norm['AcceptedCmp4']-min(dataset_norm['AcceptedCmp4']))/(max(dataset_norm['AcceptedCmp4'])-min(dataset_norm['AcceptedCmp4']))
dataset_norm['AcceptedCmp5'] = (dataset_norm['AcceptedCmp5']-min(dataset_norm['AcceptedCmp5']))/(max(dataset_norm['AcceptedCmp5'])-min(dataset_norm['AcceptedCmp5']))
dataset_norm['Complain'] = (dataset_norm['Complain']-min(dataset_norm['Complain']))/(max(dataset_norm['Complain'])-min(dataset_norm['Complain']))
dataset_norm['Educ_Levels'] = (dataset_norm['Educ_Levels']-min(dataset_norm['Educ_Levels']))/(max(dataset_norm['Educ_Levels'])-min(dataset_norm['Educ_Levels']))
dataset_norm['Marital_Levels_Binary'] = (dataset_norm['Marital_Levels_Binary']-min(dataset_norm['Marital_Levels_Binary']))/(max(dataset_norm['Marital_Levels_Binary'])-min(dataset_norm['Marital_Levels_Binary']))
dataset_norm['Purchases_Cmp_Binary'] = (dataset_norm['Purchases_Cmp_Binary']-min(dataset_norm['Purchases_Cmp_Binary']))/(max(dataset_norm['Purchases_Cmp_Binary'])-min(dataset_norm['Purchases_Cmp_Binary']))
dataset_norm['Purchases_Cmp'] = (dataset_norm['Purchases_Cmp']-min(dataset_norm['Purchases_Cmp']))/(max(dataset_norm['Purchases_Cmp'])-min(dataset_norm['Purchases_Cmp']))
dataset_norm['Mnt_Total'] = (dataset_norm['Mnt_Total']-min(dataset_norm['Mnt_Total']))/(max(dataset_norm['Mnt_Total'])-min(dataset_norm['Mnt_Total']))
dataset_norm['Total_Num_Purchases'] = (dataset_norm['Total_Num_Purchases']-min(dataset_norm['Total_Num_Purchases']))/(max(dataset_norm['Total_Num_Purchases'])-min(dataset_norm['Total_Num_Purchases']))
dataset_norm['Mnt_Regular'] = (dataset_norm['Mnt_Regular']-min(dataset_norm['Mnt_Regular']))/(max(dataset_norm['Mnt_Regular'])-min(dataset_norm['Mnt_Regular']))
dataset_norm['Family_Dependents'] = (dataset_norm['Family_Dependents']-min(dataset_norm['Family_Dependents']))/(max(dataset_norm['Family_Dependents'])-min(dataset_norm['Family_Dependents']))
dataset_norm['Seniority_Years'] = (dataset_norm['Seniority_Years']-min(dataset_norm['Seniority_Years']))/(max(dataset_norm['Seniority_Years'])-min(dataset_norm['Seniority_Years']))
dataset_norm['Seniority_Months'] = (dataset_norm['Seniority_Months']-min(dataset_norm['Seniority_Months']))/(max(dataset_norm['Seniority_Months'])-min(dataset_norm['Seniority_Months']))
dataset_norm['Age'] = (dataset_norm['Age']-min(dataset_norm['Age']))/(max(dataset_norm['Age'])-min(dataset_norm['Age']))
dataset_norm['R_Mnt_Frq'] = (dataset_norm['R_Mnt_Frq']-min(dataset_norm['R_Mnt_Frq']))/(max(dataset_norm['R_Mnt_Frq'])-min(dataset_norm['R_Mnt_Frq']))
dataset_norm['Family_Dependents_Levels'] = (dataset_norm['Family_Dependents_Levels']-min(dataset_norm['Family_Dependents_Levels']))/(max(dataset_norm['Family_Dependents_Levels'])-min(dataset_norm['Family_Dependents_Levels']))
dataset_norm['Income_Per_Income_Holders'] = (dataset_norm['Income_Per_Income_Holders']-min(dataset_norm['Income_Per_Income_Holders']))/(max(dataset_norm['Income_Per_Income_Holders'])-min(dataset_norm['Income_Per_Income_Holders']))
dataset_norm['Family_Household'] = (dataset_norm['Family_Household']-min(dataset_norm['Family_Household']))/(max(dataset_norm['Family_Household'])-min(dataset_norm['Family_Household']))
dataset_norm['Income_Per_Family_Household'] = (dataset_norm['Income_Per_Family_Household']-min(dataset_norm['Income_Per_Family_Household']))/(max(dataset_norm['Income_Per_Family_Household'])-min(dataset_norm['Income_Per_Family_Household']))


#
#columns_group_elements = ['Custid','Education','Marital_Status','Dt_Customer','Marital_Levels','Date_Freezed','Seniority']
#dataset_norm=dataset.drop(columns_group_elements, axis=1)
#sc=MinMaxScaler(feature_range=(0,1))
#dataset_norm=sc.fit_transform(dataset_norm)
#


#delete if exists
if os.path.isfile('6PM_data_MinMax.xlsx'):
    os.remove('6PM_data_MinMax.xlsx')
    
writer2 = pd.ExcelWriter('6PM_data_MinMax.xlsx')
dataset_norm.to_excel(writer2, 'Dataset_Norm')
writer2.save()



#delete if exists
if os.path.isfile('6PM_data_transformation.xlsx'):
    os.remove('6PM_data_transformation.xlsx')
    
writer3 = pd.ExcelWriter('6PM_data_transformation.xlsx')
dataset.to_excel(writer3, 'Dataset')
writer3.save()





