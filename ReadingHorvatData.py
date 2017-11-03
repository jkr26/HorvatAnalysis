# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
gb_params={'n_estimators':[20, 50, 100, 150], 'max_depth':[3, 5, 8, 10]}

"""
Created on Sun Oct 29 17:53:32 2017
Script to read and first-pass model Horvat's data
@author: jkr
"""
directory="/home/jkr/Documents/HorvatData/Analysis/Data for Keith and Ryan/3 Combined"
exp_list=list()
data_dict={}
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        exp_num=filename[filename.find('.csv')-3:filename.find('.csv')]
        if exp_num not in exp_list:
            exp_list.append(exp_num)
            data_dict[exp_num]={}
    if filename.startswith("control"):
        data_dict[exp_num]['control']=pd.read_csv(directory+"/"+filename)
    elif filename.startswith("join"):
        data_dict[exp_num]['exp']=pd.read_csv(directory+"/"+filename)[['plate_id', 'replicate', 'value', 'treatment_group', 'treatment_value']]

complete_data=pd.DataFrame()

for key in data_dict:
    complete_data=pd.concat([complete_data, data_dict[key]['exp']])
    msk=np.random.rand(len(data_dict[key]['exp']))<.8
    train_x=pd.get_dummies(data_dict[key]['exp'][['plate_id', 'replicate', 'treatment_group', 'treatment_value']])[msk]
    val_x=pd.get_dummies(data_dict[key]['exp'][['plate_id', 'replicate',  'treatment_group', 'treatment_value']])[~msk]
    train_y=data_dict[key]['exp']['value'][msk]
    val_y=data_dict[key]['exp']['value'][~msk]
    gb=GridSearchCV(GradientBoostingRegressor(), gb_params).fit(train_x, train_y)
#    ridge=RidgeCV().fit(train_x, train_y)
    score=gb.score(val_x, val_y)
#    score=ridge.score(val_x, val_y)
    print("Score for "+key+" is "+str(score))
    
msk=np.random.rand(len(complete_data))<.8
train_x=pd.get_dummies(complete_data[['plate_id', 'replicate', 'treatment_group', 'treatment_value']])[msk]
val_x=pd.get_dummies(complete_data[['plate_id', 'replicate',  'treatment_group', 'treatment_value']])[~msk]
train_y=complete_data['value'][msk]
val_y=complete_data['value'][~msk]
gb=GridSearchCV(GradientBoostingRegressor(), gb_params).fit(train_x, train_y)
#    ridge=RidgeCV().fit(train_x, train_y)
score=gb.score(val_x, val_y)
#    score=ridge.score(val_x, val_y)
print("Score for "+key+" is "+str(score))