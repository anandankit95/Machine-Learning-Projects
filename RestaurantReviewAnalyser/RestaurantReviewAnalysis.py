#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:02:53 2017

@author: Ankit
"""
#PROJECT
#Restaurant review analyser


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#here quoting =3 ignores the double quotes in the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)


#Now clean the dataset
#Remove unecessary words like - this ,and , the 
#Remove the punctuation 
#Remove the Numbers
#retain the letters
#Now convert all captial letter to  lowercase
import re
import nltk
#nltk.download('stopwords') #stopwords is a list containing unecessary words to remove
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[] #this will contain all the reviews preprocessed by nltk libraries
for i in range(dataset['Review'].size):
    review= re.sub('[^a-zA-Z]',' ',dataset['Review'][i]).lower()
    review=review.split()
    ps=PorterStemmer()  #this is used for word stemming
    #ps.stem(ite)
    review=[ps.stem(word) for word in review if  word not in set(stopwords.words('english'))]
    review= ' '.join(review) #join the words together in the obtained list
    corpus.append(review)
    
    
#create a bag of words model

#This creates the sparse matrix 
#it takes each word as a column vector
#then determines if that word is present in the current review(row)
#this creates a matrix with lot of feature vectors (ALL WORDS) having mostly 0's (Sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer
#max_features only includes 1500 most frquenctly occuring words
cv = CountVectorizer(max_features=1500) 
X= cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,-1].values 

#Split the testing and Training Data
#Use test data to classify reviews
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fit Various Machine Learning Models and Check Their Accuracy
#Fit the Naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#gives best classification accuracy of 0.673657


#Fit the SVM  Model
from sklearn.svm import SVC
classifier = SVC(C=10,kernel='rbf',random_state=0,gamma=0.2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
#gives best classification accuracy of 0.772499


#Fit the Random forest Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=1000,criterion='gini',random_state=0)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
#gives best accuracy of 0.75375


#Fit the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
#gives best accuracy 0.7050355 


#Analyze Results Via Confusion
#Create a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#calculate the f1 score
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)




#obtain the accuracy score
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()


#Parameter tunning by applying K-fold Cross Validation using Grid Search
#Applying gridSearchCV for random forest
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['gini']},
              {'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy']}]
#also add n_jobs parameter = -1 if having large dataset
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#Applying gridSearchCV for Decision tree
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [1, 10, 100, 1000], 'criterion': ['gini']},
              {'n_estimators': [1, 10, 100, 1000], 'criterion': ['entropy']}]
#also add n_jobs parameter = -1 if having large dataset
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Applying gridSearchCV for SVC
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#also add n_jobs parameter = -1 if having large dataset
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Based on above Accuracy Scores We rank Models as follows
#1 -- Support Vector Machine(SVM)
#2 -- Random Forest 
#3 -- Decision Tree
#4 -- Naive Bayes












 







