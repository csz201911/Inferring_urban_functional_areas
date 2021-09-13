# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:00:54 2021

@author: Administrator
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
from imblearn.over_sampling import SMOTE

def print_best_score(gsearch,param_test):
     # Output the best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # What parameters are used to output the best classifier
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
# 导入数据
data_train = pd.read_csv('F:\\data.csv', encoding='gbk')
data_train.index = data_train['GridID']
data_train = data_train.drop(['GridID'],axis=1)
X_train = data_train[['POInum','POItype','residentia','commercial','industry','public','greenspace', 'B2_mean',
        'B3_mean', 'B4_mean', 'B8_mean', 'B11_mean', 'B2_ent', 'B2_corr',
        'B2_asm', 'B3_ent', 'B3_corr', 'B3_asm', 'B4_ent', 'B4_corr', 'B4_asm',
        'NDBI_mean', 'NDVI_mean','workc0','worka1','worka2','worka3','worka4','worka5','workb1','workb2',
        'workb3','workb4','workb5','wendc0','wenda1','wenda2','wendb1','wendb2']]
Y_train = data_train['type60']
model_smote=SMOTE()   
x_smote_resampled,y_smote_resampled=model_smote.fit_resample(X_train,Y_train)

RF = RandomForestClassifier(max_depth=6) 
param_test_RF = {
        'max_features': range(2,25,1),
        'n_estimators': range(30,70,1),
    }
train_results = []
num_folds = 3
kfold = StratifiedKFold(n_splits=num_folds, shuffle = True, random_state= 7)
gs = GridSearchCV(estimator = RF, param_grid = param_test_RF, scoring='f1_macro', cv = kfold)

gs.fit(x_smote_resampled,y_smote_resampled)
gs.cv_results_, gs.best_params_, gs.best_score_
print_best_score(gs,param_test_RF)

CVscores = gs.cv_results_['mean_test_score']

# The optimal classifier on the training set is obtained
best_params = gs.best_params_
best_model = gs.best_estimator_

#Save the optimal model
model_file = 'finalized_RFmodel.sav'
with open(model_file, 'wb') as f:
    dump(best_model, f)

