import xgboost as xgb
import pandas as pd
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
