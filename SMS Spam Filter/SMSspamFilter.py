#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:56:14 2017

@author: Ankit
"""
#PROJECT
#SMS Spam Classifier

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('SMSSpamCollection',delimiter='\t',quoting=3)

#Now clean the dataset
#Remove unecessary words like - this ,and , the 
#Remove the punctuation 
#Remove the Numbers
#retain the letters
#Now convert all captial letter to  lowercase

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[] #this will contain all the messages preprocessed by nltk libraries
for i in range(dataset['Message'].size):
    #extract only alphabets and then convert to lowercase
    message= re.sub('[^a-zA-Z]',' ',dataset['Message'][i]).lower()
    message=message.split()
    ps=PorterStemmer()  #this is used for word stemming
    
    message=[ps.stem(word) for word in message if  word not in set(stopwords.words('english'))]
    message= ' '.join(message) #join the words together in the obtained list
    corpus.append(message)
    
#Corpus now has a list of words which are stemmed and processed by nltk modules above

#create a bag of words model
#This creates the sparse matrix 
#it takes each word as a column vector
#then determines if that word is present in the current message(row)
#this creates a matrix with lot of feature vectors (ALL WORDS) having mostly 0's (Sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer
#max_features only includes 1500 most frquenctly occuring words
cv = CountVectorizer(max_features=2000) 
X= cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,0].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Fit the Naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#Gives best accuracy of 85.40028%


#Fit the SVM  Model
from sklearn.svm import SVC
classifier = SVC(C=10,kernel='rbf',random_state=0,gamma=0.2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#Gives best accuracy of 95.1533%

#Fit the Random forest Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1000,criterion='gini',random_state=0)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
#gives best accuracy of 98.11%

#Fit the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)

#obtain the accuracy score
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10,n_jobs=-1)
accuracies.mean()
#gives best accuracy of 96.7037%



#Getting results from training model we Notice that
#1. Random forest classifier has best accuracy of 98.11% 
#Accuracy Ranking are as follows

#Random Forest --1
#Decision tree --2
#SVM           --3
#Naive Bayes   --4


     
