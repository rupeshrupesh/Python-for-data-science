#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 13:55:37 2022

@author: rupeshr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("/home/rupeshr/Desktop/TSA_Python/splitting_dataset/example1.csv")

df.shape

df.info()

df.isnull().sum()

df['Gender'].value_counts()

# replacing the missing values
df['Gender']=df['Gender'].fillna('Male')

#date extraction part
df['Start date']= pd.to_datetime(df['Start date'])
df['day']=df['Start date'].dt.day
df['month']=df['Start date'].dt.month
df['year']=df['Start date'].dt.year

# droping the meaningless feature
df.drop('Start date', axis=1, inplace=True)

# converting categorical to numerical data
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
df['Gender']=label_encoder.fit_transform(df['Gender'])

df['End Location']=label_encoder.fit_transform(df['End Location'])

df['Member type']=label_encoder.fit_transform(df['Member type'])

df['Bike number']=label_encoder.fit_transform(df['Bike number'])

corr = df.corr() #duration & member type is having some moderate positive correlation r=0.34


#plot the data based on correlation analysis
x=df['Hour'].values
y=df['Duration'].values
plt.plot(x,y)
plt.legend()


x=df.drop('Member type', axis=1) #independent variable

y=df['Member type'] #dependent varibale

# train test ratio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier   
from sklearn.metrics import confusion_matrix


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# feature scaling 
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

'''#linear model ( Logistics Regression)

classifier=LogisticRegression()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])

#KNN 
classifier2= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier2.fit(x_train, y_train)  

y_pred2=classifier2.predict(x_test)

cm2=confusion_matrix(y_test,y_pred2)

accuracy2=(cm2[0,0]+cm2[1,1])/(cm2[0,1]+cm2[1,0]+cm2[0,0]+cm2[1,1])'''


# random forecast classifier
classifier4= RandomForestClassifier(n_estimators= 10, criterion="entropy")  
classifier4.fit(x_train, y_train)  

y_pred4= classifier4.predict(x_test)

cm4=confusion_matrix(y_test,y_pred4)

accuracy4=(cm4[0,0]+cm4[1,1])/(cm4[0,1]+cm4[1,0]+cm4[0,0]+cm4[1,1])
