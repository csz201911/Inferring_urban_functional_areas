# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:00:54 2021

@author: Administrator
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
# Import data
data_train = pd.read_csv('F:\\data.csv', encoding='gbk')
data_train.index = data_train['GridID']
data_train = data_train.drop(['GridID'],axis=1)
X_train = data_train[['POInum','POItype','residentia','commercial','industry','public','greenspace', 'B2_mean',
        'B3_mean', 'B4_mean', 'B8_mean', 'B11_mean', 'B2_ent', 'B2_corr',
        'B2_asm', 'B3_ent', 'B3_corr', 'B3_asm', 'B4_ent', 'B4_corr', 'B4_asm',
        'NDBI_mean', 'NDVI_mean','workc0','worka1','worka2','worka3','worka4','worka5','workb1','workb2',
        'workb3','workb4','workb5','wendc0','wenda1','wenda2','wendb1','wendb2']]
Y_train = data_train['type60']
#Create smote model object
model_smote=SMOTE()    
x_smote_resampled,y_smote_resampled=model_smote.fit_resample(X_train,Y_train)

X_train_S1 = x_smote_resampled[['B2_mean','B3_mean', 'B4_mean', 'B8_mean', 'B11_mean', 'B2_ent', 'B2_corr',
        'B2_asm', 'B3_ent', 'B3_corr', 'B3_asm', 'B4_ent', 'B4_corr', 'B4_asm', 'NDBI_mean', 'NDVI_mean']]
X_train_S2 = x_smote_resampled[['POInum','POItype','residentia','commercial','industry','public','greenspace']]
X_train_S3 = x_smote_resampled[['workc0','worka1','worka2','worka3','worka4','worka5','workb1','workb2','workb3',
                      'workb4','workb5','wendc0','wenda1','wenda2','wendb1','wendb2']]
X_train_S5 = x_smote_resampled[['B2_mean','B3_mean', 'B4_mean', 'B8_mean', 'B11_mean', 'B2_ent', 'B2_corr',
        'B2_asm', 'B3_ent', 'B3_corr', 'B3_asm', 'B4_ent', 'B4_corr', 'B4_asm', 'NDBI_mean', 'NDVI_mean',
        'POInum','POItype','residentia','commercial','industry','public','greenspace']]
X_train_S6 = x_smote_resampled[['B2_mean','B3_mean', 'B4_mean', 'B8_mean', 'B11_mean', 'B2_ent', 'B2_corr',
        'B2_asm', 'B3_ent', 'B3_corr', 'B3_asm', 'B4_ent', 'B4_corr', 'B4_asm','NDBI_mean', 'NDVI_mean',
        'workc0','worka1','worka2','worka3','worka4','worka5','workb1','workb2','workb3','workb4',
        'workb5','wendc0','wenda1','wenda2','wendb1','wendb2']]
X_train_S4 = x_smote_resampled[['POInum','POItype','residentia','commercial','industry','public','greenspace',
                       'workc0','worka1','worka2','worka3','worka4','worka5','workb1','workb2','workb3',
                       'workb4','workb5','wendc0','wenda1','wenda2','wendb1','wendb2']]
X_train_S7 = x_smote_resampled[['POInum','POItype','residentia','commercial','industry','public','greenspace', 'B2_mean',
        'B3_mean', 'B4_mean', 'B8_mean', 'B11_mean', 'B2_ent', 'B2_corr',
        'B2_asm', 'B3_ent', 'B3_corr', 'B3_asm', 'B4_ent', 'B4_corr', 'B4_asm',
        'NDBI_mean', 'NDVI_mean', 'workc0','worka1','worka2','worka3','worka4',
        'worka5','workb1','workb2','workb3','workb4','workb5','wendc0','wenda1','wenda2','wendb1','wendb2']]

train_results = []
num_folds = 3
kfold = StratifiedKFold(n_splits=num_folds, shuffle = True, random_state=7)
#---------S1-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S1, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]  
train_results.append(results)
#---------S2-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S2, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]    
train_results.append(results)
#---------S3-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S3, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]   
train_results.append(results)
#---------S4-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S4, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]   
train_results.append(results)
#---------S5-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S5, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]    
train_results.append(results)
#---------S6-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S6, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]   
train_results.append(results)
#---------S7-------------
results = np.zeros((150,), dtype = float)
for i in range(0,50):
    model = RandomForestClassifier()
    scores = cross_val_score(model,X_train_S7, y_smote_resampled,cv=kfold,scoring='f1_macro')
    results[3*i] = scores[0]
    results[3*i+1] = scores[1]
    results[3*i+2] = scores[2]    
train_results.append(results)
# Display the results as a graph
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
fig = plt.figure()
# fig.suptitle('Accuracy of differernt combinations')
ax = fig.add_subplot(111)
plt.boxplot(train_results)
ax.set_xticklabels(['S1','S2', 'S3', 'S4','S5', 'S6', 'S7'])
plt.xlabel('Combinations',font1)
plt.ylabel('F1_macro',font1)
plt.show()