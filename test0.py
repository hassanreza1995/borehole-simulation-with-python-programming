# -*- coding: utf-8 -*-
"""
Created on Fri May  7 05:40:08 2021

@author: HassanReza
"""


from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import matplotlib.pyplot as plt  
from sklearn.cluster import MeanShift 


from sklearn.ensemble import RandomForestRegressor
from sklearn import pipeline

MainD=read_csv("D:/Work/TestTrain.csv")

MainD.describe()

MainD.var()

MainD.corr()

AK05=read_csv("D:/Work/AK05.csv")
AK16=read_csv("D:/Work/AK16.csv")
AK71=read_csv("D:/Work/AK71.csv")
Valid=read_csv("D:/Work/Valid.csv")


df=pd.DataFrame(MainD[["X","Y","Z","AlterCode","Fe"]])

ClearData=df

MinMaxModel=pp.MinMaxScaler()
MinMaxModel.fit(ClearData)
MinMaxData=MinMaxModel.transform(ClearData)

MinMaxData

AK05minmax=MinMaxModel.transform(AK05[["X","Y","Z","AlterCode","Fe"]])
AK16minmax=MinMaxModel.transform(AK16[["X","Y","Z","AlterCode","Fe"]])
AK71minmax=MinMaxModel.transform(AK71[["X","Y","Z","AlterCode","Fe"]])
Validminmax=MinMaxModel.transform(Valid[["X","Y","Z","AlterCode","Fe"]])

xAK05=AK05minmax[:,[0,1,2]]
yAK05=AK05minmax[:,[4]]

xAK16=AK16minmax[:,[0,1,2]]
yAK16=AK16minmax[:,[4]]

xAK71=AK71minmax[:,[0,1,2]]
yAK71=AK71minmax[:,[4]]

xvalid=Validminmax[:,[0,1,2]]
yvalid=Validminmax[:,[4]]


####


kmxp = ms.predict(xxfe)


x4=xxfe#MinMaxData[:,[0,1,2]]
y4=kmxp#MinMaxData[:,[4]]
x_train4, x_test4, y_train4, y_test4 = train_test_split(x4,y4, test_size= 0.2, random_state=17,stratify=y4) 
y_train4.shape=(27795, )

#model4 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
 #                         ('lr',RandomForestRegressor( random_state=17,n_estimators=100))])
model4 =RandomForestRegressor( random_state=17,n_estimators=100,criterion='mae')


model4.fit(x_train4,y_train4)
print(model4.score(x_test4,y_test4))
print(metrics.mean_squared_error(y_test4, model4.predict(x_test4)))

####4


xxfe=MinMaxData[:,[0,4]]
ms = MeanShift(bandwidth=0.145) 
ms.fit(xxfe) 
labels = ms.labels_ 
cluster_center = ms.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 
print('Number of estimated cluster:' ,n_cluster) 
 
 
#plt.scatter(xxfe[:,0], xxfe[:,1], c=labels) 
#plt.ylabel("Fe")
#plt.xlabel("X")
#plt.show() 

kmxp = ms.predict(xxfe)

kmxp.shape=(34744,1)
MinMaxData=np.append(MinMaxData,kmxp,axis=1)
#pd.DataFrame( MinMaxData).corr()

###5

xyfe=MinMaxData[:,[1,4]]
msyfe = MeanShift(bandwidth=0.12) 
msyfe.fit(xyfe) 
labels = msyfe.labels_ 
cluster_center = msyfe.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 

#plt.scatter(xxfe[:,0], xxfe[:,1], c=labels) 
#plt.ylabel("Fe")
#plt.xlabel("Y")
#plt.show() 

print('Number of estimated cluster:' ,n_cluster) 
 
kmyp = msyfe.predict(xyfe)

kmyp.shape=(34744,1)

MinMaxData=np.append(MinMaxData,kmyp,axis=1)
#pd.DataFrame( MinMaxData).corr()

###6

xzfe=MinMaxData[:,[2,4]]
mszfe = MeanShift(bandwidth=0.11) 
mszfe.fit(xzfe) 
labels = mszfe.labels_ 
cluster_center = mszfe.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 

#plt.scatter(xxfe[:,0], xxfe[:,1], c=labels) 
#plt.ylabel("Fe")
#plt.xlabel("Z")
#plt.legend()
#plt.show() 

print('Number of estimated cluster:' ,n_cluster) 
 
kmzp = mszfe.predict(xzfe)

kmzp.shape=(34744,1)

MinMaxData=np.append(MinMaxData,kmzp,axis=1)
#pd.DataFrame( MinMaxData).corr()
 
###7

xxyfe=MinMaxData[:,[0,1,4]]

msxyfe = MeanShift(bandwidth=0.172) 
msxyfe.fit(xxyfe) 
labels = msxyfe.labels_ 
cluster_center = msxyfe.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 
print('Number of estimated cluster:' ,n_cluster) 

kmxyp = msxyfe.predict(xxyfe)
kmxyp.shape=(34744,1)

MinMaxData=np.append(MinMaxData,kmxyp,axis=1)
#pd.DataFrame( MinMaxData).corr()
 
###8

xxzfe=MinMaxData[:,[0,2,4]]

msxzfe = MeanShift(bandwidth=0.19) 
msxzfe.fit(xxzfe) 
labels = msxzfe.labels_ 
cluster_center = msxzfe.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 
print('Number of estimated cluster:' ,n_cluster) 

kmxzp = msxzfe.predict(xxzfe)
kmxzp.shape=(34744,1)

MinMaxData=np.append(MinMaxData,kmxzp,axis=1)
#pd.DataFrame( MinMaxData).corr()

###9

xyzfe=MinMaxData[:,[1,2,4]]

msyzfe = MeanShift(bandwidth=0.19) 
msyzfe.fit(xyzfe) 
labels = msyzfe.labels_ 
cluster_center = msxzfe.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 
print('Number of estimated cluster:' ,n_cluster) 

kmyzp = msyzfe.predict(xyzfe)
kmyzp.shape=(34744,1)

MinMaxData=np.append(MinMaxData,kmyzp,axis=1)
#pd.DataFrame( MinMaxData).corr()

###10

xxyzfe=MinMaxData[:,[0,1,2,4]]

msxyzfe = MeanShift(bandwidth=0.206) 
msxyzfe.fit(xxyzfe) 
labels = msxyzfe.labels_ 
cluster_center = msxyzfe.cluster_centers_ 
n_cluster = len(np.unique(labels)) 
 
print('Number of estimated cluster:' ,n_cluster) 

kmxyzp = msxyzfe.predict(xxyzfe)
kmxyzp.shape=(34744,1)

MinMaxData=np.append(MinMaxData,kmxyzp,axis=1)
pd.DataFrame( MinMaxData).corr()

###
realdata=MinMaxData


#MinMaxModellast=pp.MinMaxScaler()
#MinMaxModellast.fit(MinMaxData)
#MinMaxData=MinMaxModellast.transform(MinMaxData)

MinMaxData=realdata


#plt.figure(figsize=(8,8))
#sns.heatmap(pd.DataFrame( MinMaxData).corr(),annot=True,square=True,fmt='.3f'
 #           ,linewidths=0.1,xticklabels=["X","Y","Z","Fe",1,2,3,4,5,6,7],
  #          yticklabels=["X","Y","Z","Fe",1,2,3,4,5,6,7])

x4=MinMaxData[:,[0,1,2]]
y4=MinMaxData[:,[5]]
x_train4, x_test4, y_train4, y_test4 = train_test_split(x4,y4, test_size= 0.2, random_state=17,stratify=y4) 
y_train4.shape=(27795, )

#model4 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
 #                         ('lr',RandomForestRegressor( random_state=17,n_estimators=100))])
model4 =RandomForestRegressor( random_state=17,n_estimators=100)


model4.fit(x_train4,y_train4)

print(model4.score(x_train4,y_train4))
print(model4.score(x_test4,y_test4))
print(metrics.mean_squared_error(y_test4, model4.predict(x_test4)))


x5=MinMaxData[:,[0,1,2]]   
y5=MinMaxData[:,[6]]
x_train5, x_test5, y_train5, y_test5 = train_test_split(x5,y5, test_size= 0.2, random_state=17,stratify=y5) 
y_train5.shape=(27795, )
#model5 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
#                          ('lr',RandomForestRegressor( random_state=17,n_estimators=100))])
model5 =RandomForestRegressor( random_state=17,n_estimators=100)

model5.fit(x_train5,y_train5)
print(model5.score(x_test5,y_test5))
print(metrics.mean_squared_error(y_test5, model5.predict(x_test5)))


x6=MinMaxData[:,[0,1,2]]
y6=MinMaxData[:,[7]]
x_train6, x_test6, y_train6, y_test6= train_test_split(x6,y6, test_size= 0.2, random_state=17,stratify=y6) 
y_train6.shape=(27795, )
#model6 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
#                          ('lr',RandomForestRegressor( random_state=17,n_estimators=100))])
model6 =RandomForestRegressor( random_state=17,n_estimators=100)
        
model6.fit(x_train6,y_train6)
print(model6.score(x_test6,y_test6))
print(metrics.mean_squared_error(y_test6, model6.predict(x_test6)))


x7=MinMaxData[:,[0,1,2]]
y7=MinMaxData[:,[8]]
x_train7, x_test7, y_train7, y_test7= train_test_split(x7,y7, test_size= 0.2, random_state=17,stratify=y7) 
y_train7.shape=(27795, )
#model7 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
#                         ('lr',RandomForestRegressor( random_state=17,n_estimators=1000))])
model7 =RandomForestRegressor( random_state=17,n_estimators=100)
model7.fit(x_train7,y_train7)
print(model7.score(x_test7,y_test7))
print(metrics.mean_squared_error(y_test7, model7.predict(x_test7)))


x8=MinMaxData[:,[0,1,2]]
y8=MinMaxData[:,[9]]
x_train8, x_test8, y_train8, y_test8= train_test_split(x8,y8, test_size= 0.2, random_state=17,stratify=y8) 
y_train8.shape=(27795, )
#model8 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
#                         ('lr',RandomForestRegressor( random_state=17,n_estimators=1000))])
model8 =RandomForestRegressor( random_state=17,n_estimators=100)

model8.fit(x_train8,y_train8)
print(model8.score(x_test8,y_test8))
print(metrics.mean_squared_error(y_test8, model8.predict(x_test8)))


x9=MinMaxData[:,[0,1,2]]
y9=MinMaxData[:,[10]]
x_train9, x_test9, y_train9, y_test9= train_test_split(x9,y9, test_size= 0.2, random_state=17,stratify=y9) 
y_train9.shape=(27795, )
#model9 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
#                          ('lr',RandomForestRegressor( random_state=17,n_estimators=1000))])
model9 =RandomForestRegressor( random_state=17,n_estimators=100)

model9.fit(x_train9,y_train9)
print(model9.score(x_test9,y_test9))
print(metrics.mean_squared_error(y_test9, model9.predict(x_test9)))


x10=MinMaxData[:,[0,1,2]]
y10=MinMaxData[:,[11]]
x_train10, x_test10, y_train10, y_test10= train_test_split(x10,y10, test_size= 0.2, random_state=17,stratify=y10) 
y_train10.shape=(27795, )
#model10 = pipeline.Pipeline([('rhl', RandomLayer(random_state=17, activation_func='gaussian')),
#                          ('lr',RandomForestRegressor( random_state=17,n_estimators=1000))])
model10 =RandomForestRegressor( random_state=17,n_estimators=100)

model10.fit(x_train10,y_train10)
print(model10.score(x_test10,y_test10))
print(metrics.mean_squared_error(y_test10, model10.predict(x_test10)))


import pickle

filename="D:/work/1_xyz/model4.sav"
pickle.dump(model4, open(filename, 'wb'))

filename="D:/work/1_xyz/model5.sav"
pickle.dump(model5, open(filename, 'wb'))

filename="D:/work/1_xyz/model6.sav"
pickle.dump(model6, open(filename, 'wb'))

filename="D:/work/1_xyz/model7.sav"
pickle.dump(model7, open(filename, 'wb'))

filename="D:/work/1_xyz/model8.sav"
pickle.dump(model8, open(filename, 'wb'))

filename="D:/work/1_xyz/model9.sav"
pickle.dump(model9, open(filename, 'wb'))

filename="D:/work/1_xyz/model10.sav"
pickle.dump(model10, open(filename, 'wb'))

filename="D:/work/tbr.sav"
pickle.dump(tbr, open(filename, 'wb'))

filename="D:/work/tbr.sav"
pickle.dump(tbr, open(filename, 'wb'))

filename="D:/work/MinMaxModelAll.sav"
pickle.dump(MinMaxModellast, open(filename, 'wb'))

filename="D:/work/1_xyz/MinMaxModel.sav"
pickle.dump(MinMaxModel, open(filename, 'wb'))

reg = pickle.load(open(filename, 'rb'))







