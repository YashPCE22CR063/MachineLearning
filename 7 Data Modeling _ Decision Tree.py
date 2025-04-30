# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:37:41 2021

@author: Amit Pandey
"""
#import xgboost as xgb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier


Data1 = pd.read_csv("Osteosarcoma _ with Tumor Size.csv", na_values=["Unknown"])
Test1 = Data1.copy()


###############################################################################
#Removing Null Valued Records

Test1.isnull().sum()
missing=Test1[Test1.isnull().any(axis=1)]
Test2=Test1.dropna(axis=0)

###############################################################################
#Converting Categorical Variables

Test5=pd.get_dummies(Test2, drop_first=True)

#Storing columns names
columns_list5= list(Test5.columns)

#Seprating Class names from List
features= list(set(columns_list5) - set(['Prediction']))

#Storing Output Values in Y
Y5=Test5['Prediction'].values

#Storing Output Values in x
X= Test5[features].values

#Filtering the Best Attributes
Test15 = SelectKBest(chi2, k=21).fit_transform(X, Y5)

#Splitting Training Testing data
trainX, testX, trainY5, testY5 = train_test_split(Test15,Y5,test_size=0.3,random_state=3)

###############################################################################
# Applying Decision Tree Training
###############################################################################
modelDT=DecisionTreeClassifier()
modelDT.fit(trainX,trainY5)

#Predicting from Test Data Set
predictionDT = modelDT.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixDT = confusion_matrix(testY5, predictionDT)
print(confusion_matrixDT)

#Calculating Accuracy
accuracy_scoreDT = accuracy_score(testY5, predictionDT)
precision_scoreDT = precision_score(testY5, predictionDT)
recall_scoreDT = recall_score(testY5, predictionDT)

print("precision_scoreDT:", precision_scoreDT)
print("recall_scoreDT:", recall_scoreDT)
print("accuracy_scoreDT:", accuracy_scoreDT)


