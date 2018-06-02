# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:07:05 2018

@author: Junrong
"""

import pandas as pd
import sklearn
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

print("load data..")

train = pd.read_csv("Training freq 1D, OW 16, PW 1.csv")

train2 = pd.read_csv("Training freq 1D, OW 1, PW 1.csv")
train3 = pd.read_csv("Training freq 1D, OW 2, PW 1.csv")
train4 = pd.read_csv("Training freq 1D, OW 4, PW 1.csv")
train5 = pd.read_csv("Training freq 1D, OW 8, PW 1.csv")




test = pd.read_csv("Verification freq 1D, OW 16, PW 1_feature.csv")
#test["id"] = test["id"] = (test["Machine"]+"-"+test["Date"])

test2 = pd.read_csv("Verification freq 1D, OW 1, PW 1_feature.csv")
test2["id"] = test["id"] = (test["Machine"]+"-"+test["Date"])

test3 = pd.read_csv("Verification freq 1D, OW 2, PW 1_feature.csv")
test4 = pd.read_csv("Verification freq 1D, OW 4, PW 1_feature.csv")
test5 = pd.read_csv("Verification freq 1D, OW 8, PW 1_feature.csv")



#test = pd.merge(test, test2, how='left', on='id')  
print("a quick view on the input data...")
print(train.info())
print(test.info())

#optional
#other files info
#print("other info...")

#print(train2.info())
#print(train3.info())
#print(train4.info())
#print(train5.info())


#print(test2.info())
#print(test3.info())
#print(test4.info())
#print(test5.info())



print("cleans the data...")

#fill in the null value with the mean value of this column
train["count.1"] = train["count.1"].fillna(train["count.1"].mean())
test["count.1"] = test["count.1"].fillna(test["count.1"].mean())

train["vMean.1"] = train["vMean.1"].fillna(train["vMean.1"].mean())
test["vMean.1"] = test["vMean.1"].fillna(test["vMean.1"].mean())

train["vStd.1"] = train["vStd.1"].fillna(train["vStd.1"].mean())
test["vStd.1"] = test["vStd.1"].fillna(test["vStd.1"].mean())


train["count.2"] = train["count.2"].fillna(train["count.2"].mean())
test["count.2"] = test["count.2"].fillna(test["count.2"].mean())

train["vMean.2"] = train["vMean.2"].fillna(train["vMean.2"].mean())
test["vMean.2"] = test["vMean.2"].fillna(test["vMean.2"].mean())

train["vStd.2"] = train["vStd.2"].fillna(train["vStd.2"].mean())
test["vStd.2"] = test["vStd.2"].fillna(test["vStd.2"].mean())


train["count.3"] = train["count.3"].fillna(train["count.3"].mean())
test["count.6"] = test["count.3"].fillna(test["count.3"].mean())

train["vMean.3"] = train["vMean.3"].fillna(train["vMean.3"].mean())
test["vMean.3"] = test["vMean.3"].fillna(test["vMean.3"].mean())

train["vStd.3"] = train["vStd.3"].fillna(train["vStd.3"].mean())
test["vStd.3"] = test["vStd.3"].fillna(test["vStd.3"].mean())


train["count.4"] = train["count.4"].fillna(train["count.4"].mean())
test["count.4"] = test["count.4"].fillna(test["count.4"].mean())

train["vMean.4"] = train["vMean.4"].fillna(train["vMean.4"].mean())
test["vMean.4"] = test["vMean.4"].fillna(test["vMean.4"].mean())

train["vStd.4"] = train["vStd.4"].fillna(train["vStd.4"].mean())
test["vStd.4"] = test["vStd.4"].fillna(test["vStd.4"].mean())


train["count.5"] = train["count.5"].fillna(train["count.5"].mean())
test["count.5"] = test["count.5"].fillna(test["count.5"].mean())

train["vMean.5"] = train["vMean.5"].fillna(train["vMean.5"].mean())
test["vMean.5"] = test["vMean.5"].fillna(test["vMean.5"].mean())

train["vStd.5"] = train["vStd.5"].fillna(train["vStd.5"].mean())
test["vStd.5"] = test["vStd.5"].fillna(test["vStd.5"].mean())


train["count.6"] = train["count.6"].fillna(train["count.6"].mean())
test["count.6"] = test["count.6"].fillna(test["count.6"].mean())

train["vMean.6"] = train["vMean.6"].fillna(train["vMean.6"].mean())
test["vMean.6"] = test["vMean.6"].fillna(test["vMean.6"].mean())

train["vStd.6"] = train["vStd.6"].fillna(train["vStd.6"].mean())
test["vStd.6"] = test["vStd.6"].fillna(test["vStd.6"].mean())


train["count.7"] = train["count.7"].fillna(train["count.7"].mean())
test["count.7"] = test["count.7"].fillna(test["count.7"].mean())

train["vMean.7"] = train["vMean.7"].fillna(train["vMean.7"].mean())
test["vMean.7"] = test["vMean.7"].fillna(test["vMean.7"].mean())

train["vStd.7"] = train["vStd.7"].fillna(train["vStd.7"].mean())
test["vStd.7"] = test["vStd.7"].fillna(test["vStd.7"].mean())


train["count.8"] = train["count.8"].fillna(train["count.8"].mean())
test["count.8"] = test["count.8"].fillna(test["count.8"].mean())

train["vMean.8"] = train["vMean.8"].fillna(train["vMean.8"].mean())
test["vMean.8"] = test["vMean.8"].fillna(test["vMean.8"].mean())

train["vStd.8"] = train["vStd.8"].fillna(train["vStd.8"].mean())
test["vStd.8"] = test["vStd.8"].fillna(test["vStd.8"].mean())


train["count.9"] = train["count.9"].fillna(train["count.9"].mean())
test["count.9"] = test["count.9"].fillna(test["count.9"].mean())

train["vMean.9"] = train["vMean.9"].fillna(train["vMean.9"].mean())
test["vMean.9"] = test["vMean.9"].fillna(test["vMean.9"].mean())

train["vStd.9"] = train["vStd.9"].fillna(train["vStd.9"].mean())
test["vStd.9"] = test["vStd.9"].fillna(test["vStd.9"].mean())


train["count.10"] = train["count.10"].fillna(train["count.10"].mean())
test["count.10"] = test["count.10"].fillna(test["count.10"].mean())

train["vMean.10"] = train["vMean.10"].fillna(train["vMean.10"].mean())
test["vMean.10"] = test["vMean.10"].fillna(test["vMean.10"].mean())

train["vStd.10"] = train["vStd.10"].fillna(train["vStd.10"].mean())
test["vStd.10"] = test["vStd.10"].fillna(test["vStd.10"].mean())


train["count.11"] = train["count.11"].fillna(train["count.11"].mean())
test["count.11"] = test["count.11"].fillna(test["count.11"].mean())

train["vMean.11"] = train["vMean.11"].fillna(train["vMean.11"].mean())
test["vMean.11"] = test["vMean.11"].fillna(test["vMean.11"].mean())

train["vStd.11"] = train["vStd.11"].fillna(train["vStd.11"].mean())
test["vStd.11"] = test["vStd.11"].fillna(test["vStd.11"].mean())


train["count.12"] = train["count.12"].fillna(train["count.12"].mean())
test["count.12"] = test["count.12"].fillna(test["count.12"].mean())

train["vMean.12"] = train["vMean.12"].fillna(train["vMean.12"].mean())
test["vMean.12"] = test["vMean.12"].fillna(test["vMean.12"].mean())

train["vStd.12"] = train["vStd.12"].fillna(train["vStd.12"].mean())
test["vStd.12"] = test["vStd.12"].fillna(test["vStd.12"].mean())


train["count.13"] = train["count.13"].fillna(train["count.13"].mean())
test["count.13"] = test["count.13"].fillna(test["count.13"].mean())

train["vMean.13"] = train["vMean.13"].fillna(train["vMean.13"].mean())
test["vMean.13"] = test["vMean.13"].fillna(test["vMean.13"].mean())

train["vStd.13"] = train["vStd.13"].fillna(train["vStd.13"].mean())
test["vStd.13"] = test["vStd.13"].fillna(test["vStd.13"].mean())


train["count.14"] = train["count.14"].fillna(train["count.14"].mean())
test["count.14"] = test["count.14"].fillna(test["count.14"].mean())

train["vMean.14"] = train["vMean.14"].fillna(train["vMean.14"].mean())
test["vMean.14"] = test["vMean.14"].fillna(test["vMean.14"].mean())

train["vStd.14"] = train["vStd.14"].fillna(train["vStd.14"].mean())
test["vStd.14"] = test["vStd.14"].fillna(test["vStd.14"].mean())


train["count.15"] = train["count.15"].fillna(train["count.15"].mean())
test["count.15"] = test["count.15"].fillna(test["count.15"].mean())

train["vMean.15"] = train["vMean.15"].fillna(train["vMean.15"].mean())
test["vMean.15"] = test["vMean.15"].fillna(test["vMean.15"].mean())

train["vStd.15"] = train["vStd.15"].fillna(train["vStd.15"].mean())
test["vStd.15"] = test["vStd.15"].fillna(test["vStd.15"].mean())


train["count.16"] = train["count.16"].fillna(train["count.16"].mean())
test["count.16"] = test["count.16"].fillna(test["count.16"].mean())

train["vMean.16"] = train["vMean.16"].fillna(train["vMean.16"].mean())
test["vMean.16"] = test["vMean.16"].fillna(test["vMean.16"].mean())

train["vStd.16"] = train["vStd.16"].fillna(train["vStd.16"].mean())
test["vStd.16"] = test["vStd.16"].fillna(test["vStd.16"].mean())


train["count.17"] = train["count.17"].fillna(train["count.17"].mean())
test["count.17"] = test["count.17"].fillna(test["count.17"].mean())

train["vMean.17"] = train["vMean.17"].fillna(train["vMean.17"].mean())
test["vMean.17"] = test["vMean.17"].fillna(test["vMean.17"].mean())

train["vStd.17"] = train["vStd.17"].fillna(train["vStd.17"].mean())
test["vStd.17"] = test["vStd.17"].fillna(test["vStd.17"].mean())


train["count.18"] = train["count.18"].fillna(train["count.18"].mean())
test["count.18"] = test["count.18"].fillna(test["count.18"].mean())

train["vMean.18"] = train["vMean.18"].fillna(train["vMean.18"].mean())
test["vMean.18"] = test["vMean.18"].fillna(test["vMean.18"].mean())

train["vStd.18"] = train["vStd.18"].fillna(train["vStd.18"].mean())
test["vStd.18"] = test["vStd.18"].fillna(test["vStd.18"].mean())


train["count.19"] = train["count.19"].fillna(train["count.19"].mean())
test["count.19"] = test["count.19"].fillna(test["count.19"].mean())

train["vMean.19"] = train["vMean.19"].fillna(train["vMean.19"].mean())
test["vMean.19"] = test["vMean.19"].fillna(test["vMean.19"].mean())

train["vStd.19"] = train["vStd.19"].fillna(train["vStd.19"].mean())
test["vStd.19"] = test["vStd.19"].fillna(test["vStd.19"].mean())


train["count.20"] = train["count.20"].fillna(train["count.20"].mean())
test["count.20"] = test["count.20"].fillna(test["count.20"].mean())

train["vMean.20"] = train["vMean.20"].fillna(train["vMean.20"].mean())
test["vMean.20"] = test["vMean.20"].fillna(test["vMean.20"].mean())

train["vStd.20"] = train["vStd.20"].fillna(train["vStd.20"].mean())
test["vStd.20"] = test["vStd.20"].fillna(test["vStd.20"].mean())


train["count"] = train["count"].fillna(train["count"].mean())
test["count"] = test["count"].fillna(test["count"].mean())

train["all_count"] = train["count"]+train["count.1"]+train["count.2"]+train["count.3"]+\
    train["count.4"]+train["count.5"]+train["count.6"]+train["count.7"]+train["count.8"]+\
    train["count.9"]+train["count.10"]+train["count.11"]+train["count.12"]+train["count.13"]+\
    train["count.14"]+train["count.15"]+train["count.16"]+train["count.17"]+train["count.18"]

test["all_count"] = test["count"]+test["count.1"]+test["count.2"]+test["count.3"]+\
    test["count.4"]+test["count.5"]+test["count.6"]+test["count.7"]+test["count.8"]+\
    test["count.9"]+test["count.10"]+test["count.11"]+test["count.12"]+test["count.13"]+\
    test["count.14"]+test["count.15"]+test["count.16"]+test["count.17"]+test["count.18"]
   
    
train["all_count"] = train["all_count"].fillna(0)
test["all_count"] = test["all_count"].fillna(0)
    
train["result"] = train["result"].fillna("TRUE")

train["result"] = train["result"].apply(lambda x : 1 if x else 0)


#date processing
#train["Date"] = pd.to_datetime(train.Date)
#test["Date"] = pd.to_datetime(test.Date)
#train["Date"] = train["Date"].dt.strftime('%d/%b/%Y')
#test["Date"] = test["Date"].dt.strftime('%d/%b/%Y')


#feature selection
feature = ["count",\
           "count.6","count.8","count.14","count.15","count.17","count.11","count.20",\
           "vMean.6","vMean.8","vMean.14","vMean.15","vMean.17","vMean.11","vMean.20",\
           "vStd.6","vStd.8","vStd.14","vStd.15","vStd.17","vStd.11","vStd.20",\
           "all_count"]

feature = ["count",\
           "count.6","count.8","count.14","count.15","count.17","count.11","count.20",\
           "vMean.6","vMean.8","vMean.14","vMean.15","vMean.17","vMean.11","vMean.20",\
           "vStd.6","vStd.8","vStd.14","vStd.15","vStd.17","vStd.11","vStd.20"]
           
feature = ["count.6","vMean.6","vStd.6",\
           "count.14","vMean.14","vStd.14",\
           "count.15","vMean.15","vStd.15",\
           "count.17","vMean.17","vStd.17",\
           "count.20","vMean.20","vStd.20"]


print("data visualization..")
train.result.value_counts().plot(kind= 'bar')


print("fix model...")

#indicator of whether we use the linear regression model
#we need to adjust the output for linear regression
linear_regression_used = 0

#we can select our model here


#decision tree 71.8
#from sklearn import tree
#model = tree.DecisionTreeClassifier()

#xgboost 76.6
#model = XGBClassifier()


#KNN 65.5
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors=20)

#random forest 49.7
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics 
model = RandomForestClassifier(n_jobs=-1, random_state=0)


#neural network 65.6
#from sklearn.neural_network import MLPClassifier
#model = MLPClassifier(hidden_layer_sizes=(10,8,6,4,2),max_iter=10000)
    
#linear regression
#from sklearn import linear_model
#model = linear_model.LinearRegression()
#linear_regression_used = 1



#SGD 62.7
#from sklearn import linear_model
#model = linear_model.SGDClassifier()

#QDA 
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#model = QuadraticDiscriminantAnalysis(reg_param = 5)

#bagging 69.7
#from sklearn.ensemble import BaggingClassifier
#from sklearn.svm import SVC
#model = SVC(kernel='rbf', degree=3)
#model =BaggingClassifier(model, n_estimators=20, max_samples=0.8, max_features=1.0,bootstrap=True, bootstrap_features=False)
       

#AdaBoost 65.5
#from sklearn.ensemble import AdaBoostClassifier
#model =AdaBoostClassifier()

#svm 50
#from sklearn import svm
#model = svm.SVC()

#advanced svm
#from sklearn.svm import SVC, LinearSVC
#model = SVC(C = 30, gamma = 0.01)

#K Cluster
#from sklearn.cluster import KMeans
#model = KMeans(n_clusters=2, random_state=0)



##################################################
model.fit(train[feature], train["result"])
predict_data = model.predict(test[feature])
predict_data_train = model.predict(train[feature])
predict_data_train_frame = pd.DataFrame({
        "result": predict_data_train
        })
#predict_data_train_frame.loc[predict_data_train_frame['result'] > 0.0773,'result'] = round(1)  
#predict_data_train_frame.loc[predict_data_train_frame['result'] <= 0.0773,'result'] = round(0)
print("training accuracy...")
print(accuracy_score(predict_data_train_frame["result"],train["result"]))
print("training accurate terms count...")
print(accuracy_score(predict_data_train_frame["result"],train["result"],normalize=False))
##################################################

'''
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
model = SVC(kernel='rbf', degree=3)

bagging_clf =BaggingClassifier(model, n_estimators=20, max_samples=0.8, max_features=1.0,bootstrap=True, bootstrap_features=False)

bagging_clf.fit(train[feature], train["result"])

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adb =AdaBoostClassifier()
adb.fit(train[feature],train["result"])

predict_data = adb.predict(test[feature])
'''



print("final wrap up...")
submission = pd.DataFrame({
        "id": (test["Machine"]+"-"+test["Date"]),
        "Label": predict_data
        })
    
#used for linear regression
if(linear_regression_used > 0.5):
    submission.loc[submission['Label'] > 0.0773,'Label'] = round(1)  
    submission.loc[submission['Label'] <= 0.0773,'Label'] = round(0)


print("write to file...")
submission.to_csv("submission.csv",columns=["id","Label"],index = False)

print("a quick view of prediction result...")
submission.Label.value_counts().plot(kind= 'bar')




