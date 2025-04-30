import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


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


"""
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

#Normalizing the dataset
scaler = MinMaxScaler()
Data_Normal_Train = scaler.fit_transform(trainX)
Data_Normal_Test = scaler.fit_transform(testX)

clf.fit(trainX, trainY5)
prediction = clf.predict(testX)

#Calculating Accuracy
accuracy_score = accuracy_score(testY5, prediction)
precision_score = precision_score(testY5, prediction)
recall_score = recall_score(testY5, prediction)

print("precision_score:", precision_score)
print("recall_score:", recall_score)
print("accuracy_score:", accuracy_score)


###############################################################################
# Applying SVM:Gaussian Kernel Training
###############################################################################
modelSVM = LinearSVC(random_state=42)
modelSVM.fit(trainX,trainY5)

# Predicting from Test Data Set
predictionSVM = modelSVM.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixSVM = confusion_matrix(testY5, predictionSVM)
print(confusion_matrixSVM)

#Calculating Accuracy
accuracy_scoreSVM = accuracy_score(testY5, predictionSVM)
precision_scoreSVM = precision_score(testY5, predictionSVM)
recall_scoreSVM = recall_score(testY5, predictionSVM)

print("precision_scoreSVM:", precision_scoreSVM)
print("recall_scoreSVM:", recall_scoreSVM)
print("accuracy_scoreSVM:", accuracy_scoreSVM)


# Applying Logistic Regression Training
###############################################################################
modelLR=LogisticRegression()
modelLR.fit(trainX,trainY5)

# Predicting from Test Data Set
predictionLR = modelLR.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixLR = confusion_matrix(testY5, predictionLR)
print(confusion_matrixLR)

#Calculating Accuracy
accuracy_scoreLR = accuracy_score(testY5, predictionLR)
precision_scoreLR = precision_score(testY5, predictionLR)
recall_scoreLR = recall_score(testY5, predictionLR)

print("precision_scoreLR:", precision_scoreLR)
print("recall_scoreLR:", recall_scoreLR)
print("accuracy_scoreLR:", accuracy_scoreLR)

# Applying Random Forest Training
###############################################################################
modelRF=RandomForestClassifier(n_estimators=10, random_state=42)
modelRF.fit(trainX,trainY5)

# Predicting from Test Data Set
predictionRF = modelRF.predict(testX)

###############################################################################
#Confusion Matrix

#Calculating Accuracy
accuracy_scoreRF = accuracy_score(testY5, predictionRF)
precision_scoreRF = precision_score(testY5, predictionRF)
recall_scoreRF = recall_score(testY5, predictionRF)

print("precision_scoreRF:", precision_scoreRF)
print("recall_scoreRF:", recall_scoreRF)
print("accuracy_scoreRF:", accuracy_scoreRF)


clf = BaggingClassifier(base_estimator=LogisticRegression(),
                        n_estimators=10, random_state=2).fit(trainX, trainY5)

prediction = clf.predict(testX)

#Calculating Accuracy
accuracy_score = accuracy_score(testY5, prediction)
precision_score = precision_score(testY5, prediction)
recall_score = recall_score(testY5, prediction)

print("precision_score:", precision_score)
print("recall_score:", recall_score)
print("accuracy_score:", accuracy_score)

"""
###############################################################################
# Applying Logistic Regression Training
###############################################################################
modelLR=LogisticRegression()
modelLR.fit(trainX,trainY5)

# Predicting from Test Data Set
predictionLR = modelLR.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixLR = confusion_matrix(testY5, predictionLR)
print(confusion_matrixLR)

#Calculating Accuracy
accuracy_scoreLR = accuracy_score(testY5, predictionLR)
precision_scoreLR = precision_score(testY5, predictionLR)
recall_scoreLR = recall_score(testY5, predictionLR)

print("precision_scoreLR:", precision_scoreLR)
print("recall_scoreLR:", recall_scoreLR)
print("accuracy_scoreLR:", accuracy_scoreLR)


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


###############################################################################
# Applying SVM:Gaussian Kernel Training
###############################################################################
modelSVM = SVC(kernel='rbf')
modelSVM.fit(trainX,trainY5)

# Predicting from Test Data Set
predictionSVM = modelSVM.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixSVM = confusion_matrix(testY5, predictionSVM)
print(confusion_matrixSVM)

#Calculating Accuracy
accuracy_scoreSVM = accuracy_score(testY5, predictionSVM)
precision_scoreSVM = precision_score(testY5, predictionSVM)
recall_scoreSVM = recall_score(testY5, predictionSVM)

print("precision_scoreSVM:", precision_scoreSVM)
print("recall_scoreSVM:", recall_scoreSVM)
print("accuracy_scoreSVM:", accuracy_scoreSVM)

###############################################################################
# Applying Naive Bayes Training
###############################################################################
modelNB = MultinomialNB()
modelNB.fit(trainX,trainY5)

# Predicting from Test Data Set
predictionNB = modelNB.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixNB = confusion_matrix(testY5, predictionNB)
print(confusion_matrixNB)

#Calculating Accuracy
accuracy_scoreNB = accuracy_score(testY5, predictionNB)
precision_scoreNB = precision_score(testY5, predictionNB)
recall_scoreNB = recall_score(testY5, predictionNB)

print("precision_scoreNB:", precision_scoreNB)
print("recall_scoreNB:", recall_scoreNB)
print("accuracy_scoreNB:", accuracy_scoreNB)

###############################################################################
# Implementing Voting Classifier
###############################################################################
clf1 = LogisticRegression()
clf2 = SVC(kernel='rbf')
clf3 = DecisionTreeClassifier()
#MultinomialNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1.fit(trainX,trainY5)

# Predicting from Test Data Set   'gnb', clf3
predictionVOT = eclf1.predict(testX)

###############################################################################
#Confusion Matrix
confusion_matrixVOT = confusion_matrix(testY5, predictionVOT)
print(confusion_matrixVOT)

#Calculating Accuracy
accuracy_scoreVOT = accuracy_score(testY5, predictionVOT)
precision_scoreVOT = precision_score(testY5, predictionVOT)
recall_scoreVOT = recall_score(testY5, predictionVOT)

print("precision_scoreVOT:", precision_scoreVOT)
print("recall_scoreVOT:", recall_scoreVOT)
print("accuracy_scoreVOT:", accuracy_scoreVOT)
