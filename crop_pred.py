AP = []
MN = []


import pandas as pd
#import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
from collections import Counterz 

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

DF0 = pd.read_csv("/content/drive/MyDrive/prediction2/Crop_recommendation.csv")

NF = [a for a in DF0.columns if DF0[a].dtypes != 'O']
print("Numerical Features Count {}".format(len(NF)))

DF0.isnull().sum()*100/len(DF0)

def RandomSamplingImputation(DF0, variable):
    DF0[variable]=DF0[variable]
    random_sample=DF0[variable].dropna().sample(DF0[variable].isnull().sum(),random_state=0)
    random_sample.index=DF0[DF0[variable].isnull()].index
    DF0.loc[DF0[variable].isnull(),variable]=random_sample

DF = [a for a in NF if len(DF0[a].unique())<25]
CF = [a for a in NF if a not in DF]

a = DF0[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
Target = DF0['label']
Labels = DF0['label']

X_train, X_test, y_train, y_test = train_test_split(a, Target, test_size =0.2, random_state = 2)

SVM = SVC(gamma='auto')
SVM.fit(X_train,y_train)

DT = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
DT.fit(X_train,y_train)

LR = LogisticRegression(random_state=2)
LR.fit(X_train,y_train)

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, y_train)

CB = CatBoostClassifier(iterations=3000, eval_metric = "AUC")
CB.fit(X_train, y_train)

GNB = GaussianNB()
GNB.fit(X_train, y_train)

XGB_C = xgb.XGBClassifier()
XGB_C.fit(X_train, y_train)


RF=RandomForestClassifier()
RF.fit(X_train,y_train)

PRED0 = SVM.predict(X_test)
x = metrics.accuracy_score(y_test,PRED0)
AP.append(x)
MN.append('SVM')
print("SVM Accuracy ==>: ", x)
print(classification_report(y_test,PRED0))

