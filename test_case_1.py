#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:16:27 2022

@author: rupeshr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("/home/rupeshr/Desktop/TSA_Python/splitting_dataset/example.csv")

df.shape

df.info()

df.isnull().sum()

df['DateTime']= pd.to_datetime(df['DateTime'])

import datetime, time

#converting datetime to ordinal format because datetime function wont work in linear regression
df['conv_date']= pd.to_datetime(df.DateTime, format="%Y-%M-%D")
df['conv_date']=df['conv_date'].apply(lambda var: time.mktime(var.timetuple()))


corr=df.corr()

df.drop(['Company Id','DateTime'], axis=1, inplace=True)


x=df.drop('fuel consumption', axis=1) #independent variable

y=df['fuel consumption'] #dependent varibale

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  


from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  

y_pred= regressor.predict(x_test)  

print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))  
