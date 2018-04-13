import pandas as pd
import sklearn.tree as tree
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

#machine learning algorithms and methods implementation here
print("load data..")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#test = train[660:891]
#train = [0:660]

#print(train.info())
#print(test.info())
#data overview
#Data columns (total 12 columns):
#PassengerId    891 non-null int64
#Survived       891 non-null int64
#Pclass         891 non-null int64
#Name           891 non-null object
#Sex            891 non-null object
#Age            714 non-null float64
#SibSp          891 non-null int64
#Parch          891 non-null int64
#Ticket         891 non-null object
#Fare           891 non-null float64
#Cabin          204 non-null object
#Embarked       889 non-null object

print("cleans the data...")

#age
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].mean())

#gender
train["Sex"] = train["Sex"].apply(lambda x : 1 if x == "male" else 0)
test["Sex"] = test["Sex"].apply(lambda x : 1 if x == "male" else 0)

#fare
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

#Parch
train["Parch"] = train["Parch"].fillna(train["Parch"].median())
test["Parch"] = test["Parch"].fillna(test["Parch"].median())

#Embarked
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

train["Embarked"] = train["Embarked"].apply(lambda x : 0 if x == "S" else 1)
test["Embarked"] = test["Embarked"].apply(lambda x : 0 if x == "S" else 1)

#Pclass
train["Pclass"] = train["Pclass"].fillna(3)
test["Pclass"] = test["Pclass"].fillna(3)

#SibSp
train["SibSp"] = train["SibSp"].fillna(train["SibSp"].mean())
test["SibSp"] = test["SibSp"].fillna(test["SibSp"].mean())


feature = ["Age","Sex","Fare","Parch","Embarked","Pclass","SibSp"]
#feature = ["Age","Sex","Fare"]


print("data visualization..")
plt.subplot(231)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
train.Survived.value_counts().plot(kind= 'bar')

plt.subplot(232)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
train.Embarked.value_counts().plot(kind= 'bar')

plt.subplot(233)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
train.Parch.value_counts().plot(kind= 'bar')

plt.subplot(234)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
train.Pclass.value_counts().plot(kind= 'bar')

plt.subplot(235)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
train.SibSp.value_counts().plot(kind= 'bar')

plt.subplot(236)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
train.Sex.value_counts().plot(kind= 'bar')


print("fix model...")

#decision tree 71.8
#dt = tree.DecisionTreeClassifier()
#dt = dt.fit(train[feature],train["Survived"])
#predict_data = dt.predict(test[feature])

#xgboost 76.6
model = XGBClassifier()
model.fit(train[feature], train["Survived"])

#KNN 65.5
#from sklearn.neighbors import KNeighborsClassifier
#model = KNeighborsClassifier(n_neighbors=20)
#model.fit(train[feature], train["Survived"])

#random forest 75.2
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import GridSearchCV
#from sklearn import cross_validation, metrics 
#model = RandomForestClassifier(n_jobs=-1, random_state=0)
#model.fit(train[feature], train["Survived"])
#predict_data = model.predict(test[feature])

#neural network 65.6
#from sklearn.neural_network import MLPClassifier
#model = MLPClassifier(hidden_layer_sizes=(10,8,6,4,2),max_iter=10000)
#model.fit(train[feature],train["Survived"])
    
#linear regression
#from sklearn import linear_model
#model = linear_model.LinearRegression()
#model.fit(train[feature], train["Survived"])
    

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

#svm 
#from sklearn import svm
#model = svm.SVC()

#advanced svm
#from sklearn.svm import SVC, LinearSVC
#model = SVC(C = 30, gamma = 0.01)
#model.fit(train[feature], train["Survived"])

#K Cluster
#from sklearn.cluster import KMeans
#model = KMeans(n_clusters=2, random_state=0)



##################################################
model.fit(train[feature], train["Survived"])
predict_data = model.predict(test[feature])
##################################################

'''
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
model = SVC(kernel='rbf', degree=3)

bagging_clf =BaggingClassifier(model, n_estimators=20, max_samples=0.8, max_features=1.0,bootstrap=True, bootstrap_features=False)

bagging_clf.fit(train[feature], train["Survived"])

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adb =AdaBoostClassifier()
adb.fit(train[feature],train["Survived"])

predict_data = adb.predict(test[feature])
'''



print("write to file...")
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_data
        })
    
#used for linear regression
#submission.loc[submission['Survived'] > 0.5,'Survived'] = round(1)  
#submission.loc[submission['Survived'] <= 0.5,'Survived'] = round(0)
    
submission.to_csv("submission.csv",index = False)
