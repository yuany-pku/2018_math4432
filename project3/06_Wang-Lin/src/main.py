# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


import os

def cabin_transform(cabin):
    if not cabin:
        return 'Z'
    else:
        return cabin[0]


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# plt.hist(x=(train_df['Survived']==1))
# plt.title('Survival Distribution in Training Set')
# plt.xlabel('Survived or not')
# plt.ylabel('#Passengers')
# plt.legend()
# plt.show()
# #visualize
#
# # print(test_df.describe())
#
#
# sns.set()
#
# plt.figure(figsize=[24,16])
#
# plt.subplot(224)
# survived_ages=[age for age in train_df[train_df['Survived']==1]['Age'] if age!=None]
# dead_ages=train_df[train_df['Survived']==0][train_df[train_df['Survived']==0]['Age'].notnull()]['Age']
# plt.hist(x = [dead_ages,survived_ages ],
#          color = ['r', 'b'],label = ['Dead','Survived'])
# plt.title('Age Histogram by Survival')
# plt.xlabel('Age (Years)')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(222)
# survived_cabins=train_df[train_df['Survived']==1]['Cabin'].notnull()
# dead_cabins=train_df[train_df['Survived']==0]['Cabin'].notnull()
# plt.hist(x = [dead_cabins,survived_cabins ],
#          color = ['r', 'b'],label = ['Dead','Survived'])
# plt.title('Has Cabin Histogram by Survival')
# plt.xlabel('HasCabin')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(223)
# survived_ages=[age for age in train_df[train_df['Survived']==1]['Sex'] if age!=None]
# dead_ages=[age for age in train_df[train_df['Survived']==0]['Sex'] if age!=None]
# plt.hist(x = [dead_ages,survived_ages ],
#          color = ['r', 'b'],label = ['Dead','Survived'])
# plt.title('Sex Distribution by Survival')
# plt.xlabel('Sex')
# plt.ylabel('# of Passengers')
# plt.legend()
#
#
# plt.subplot(221)
# survived_ages=[age for age in train_df[train_df['Survived']==1]['Pclass'] if age!=None]
# dead_ages=[age for age in train_df[train_df['Survived']==0]['Pclass'] if age!=None]
# plt.hist(x = [dead_ages,survived_ages ],
#          color = ['r', 'b'],label = ['Dead','Survived'])
# plt.title('Pclass Distribution by Survival')
# plt.xlabel('Pclass')
# plt.ylabel('# of Passengers')
# plt.legend()
#
#
#
# plt.show()
# print('finish')
#
# Cabin_fare=train_df[train_df['Cabin'].notnull()]['Fare']
# nonCabin_fare=train_df[train_df['Cabin'].isnull()]['Fare']
# plt.hist(x = [Cabin_fare,nonCabin_fare],
#          color = ['r', 'b'],label = ['Has Cabin','No Cabin'])
# plt.title('Correlation between HasCabin and fare')
# plt.xlabel('Fare')
# plt.ylabel('# of Passengers')
# plt.show()
#
#
# print(train_df.loc[train_df['Fare'].notnull(),['Fare']].iloc[:,0].unique() )
# plt.hist(train_df.loc[train_df['Fare'].notnull(),['Fare']].values , 20)
# plt.show()
#
# plt.figure(figsize=[24,16])
#
# # plt.subplot(321)
#
# # # plt.title(' Fare Distribution')
# # print('finish')
# plt.legend()
# plt.subplot(221)
# plt.hist(x = [train_df['Cabin'].notnull(), test_df['Cabin'].notnull()],
#          stacked=False, color = ['b', 'r'],label = ['Train','Test'])
# plt.title(' train/test Cabin Distribution')
# plt.xlabel('HasCabin')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(222)
# plt.hist(x = [train_df['Pclass'], test_df['Pclass']],
#          color = ['b', 'r'],label = ['Train','Test'])
# plt.title(' train/test Pclass Distribution')
# plt.xlabel('Pclass')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(223)
# plt.hist(x = [train_df['Sex'], test_df['Sex']],
#           color = ['b', 'r'],label = ['Train','Test'])
# plt.title(' train/test Sex Distribution')
# plt.xlabel('sex')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(224)
# plt.hist(x = [train_df['Age'][train_df['Age'].notnull()], test_df['Age'][test_df['Age'].notnull()]],
#          stacked=False, color = ['b', 'r'],label = ['Train','Test'])
# plt.title(' train/test Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.show()
# print('finish')
#
#




###########################
#Now start to complete, remove, create, transform
from sklearn.preprocessing import LabelEncoder

Y_train=train_df.pop('Survived')
test_passenger_id=test_df['PassengerId']
all_df=pd.concat([train_df,test_df])
all_df.drop(['PassengerId'],axis=1,inplace=True)
# print(test_passenger_id.head)


#1. first, convert sex to numeric data type
all_df['Sex']=all_df['Sex'].astype('category')
print(all_df.dtypes)
all_df['Sex']=all_df['Sex'].cat.codes


#2.add together sibsp and parch to create new column: Family, and drop the original
all_df['Family']=all_df['SibSp']+all_df['Parch']
all_df.drop(['SibSp','Parch'],axis=1,inplace=True)

guess_ages=np.zeros((2,3))
#3. fill in age group with prediction from Pclass and sex
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = all_df[(all_df['Sex'] == i) & \
                           (all_df['Pclass'] == j + 1)]['Age'].dropna()

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        all_df.loc[(all_df.Age.isnull()) & (all_df.Sex == i) & (all_df.Pclass == j + 1), \
                    'Age'] = guess_ages[i, j]

all_df['Age'] = all_df['Age'].astype(int)
lb=LabelEncoder()
# all_df['Age_groups']=lb.fit_transform(pd.cut(all_df['Age'],5))
# # all_df["Age_under15"]=all_df.Age.apply(lambda age:age<15)
# # all_df["Age_above60"]=all_df.Age.apply(lambda age:age>60)
# all_df.drop('Age', axis=1,inplace=True)

#4.categoritize cabin
all_df['Cabin']=all_df['Cabin'].astype(str).apply(cabin_transform)
# print(all_df.describe())
all_df=pd.concat([all_df,pd.get_dummies(all_df['Cabin'],drop_first=True)],axis=1)
all_df.drop(['Cabin'],axis=1,inplace=True)
# print(all_df.describe())
# print(pd.isnull(all_df).any(1).nonzero()[0])

#5. categoritize embark
# all_df=pd.concat([all_df,pd.get_dummies(all_df['Embarked'],drop_first=True)],axis=1)
all_df.drop(['Embarked'],axis=1,inplace=True)


#6. drop ticket and name, after engineering name
all_df['Title'] = all_df['Name']
# Cleaning name and extracting Title
for name_string in all_df['Name']:
    all_df['Title'] = all_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)


mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
all_df.replace({'Title': mapping}, inplace=True)

all_df=pd.concat([all_df,pd.get_dummies(all_df['Title'],drop_first=True)],axis=1)
all_df.drop(['Ticket','Name','Title'],axis=1,inplace=True)


#7. fare has one missing value in test set, predict it using the third class median
missing_index=all_df.Fare.isnull().nonzero()[0][0]
# print(missing_index)
# all_df.Fare.iloc[missing_index]= all_df.loc[((all_df['Pclass']==3) & (all_df['S']==1.0)), 'Fare'].median(axis=0)
all_df.Fare.iloc[missing_index]= all_df.loc[((all_df['Pclass']==3)), 'Fare'].median(axis=0)
# all_df['Fare']=all_df['Fare'].apply(lambda x: np.log(x+1))
# print(all_df['Fare'].unique())



#8. VERY IMPOARTANT:  COLUMN STANDARDIZATION:
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
standard_col=['Pclass','Sex','Fare','Age','Family']
flexible_col=[x for x in all_df.columns if x not in standard_col]

all_df[standard_col]=scaler.fit_transform(all_df[standard_col])
all_df[flexible_col]=all_df[flexible_col].transform(lambda column: (column-column.mean())/(column.std()/column.mean()))

print(all_df.describe())
# print(all_df.loc[((all_df['Pclass']==3) & (all_df['S']==1.0)), 'Fare'].median(axis=0))
# print(all_df.index.tolist())

# print(all_df.columns)
# for column in all_df.columns:
#     print(type(column))
#     plt.hist(all_df[column].values)
#     plt.title('distribution of column: {}'.format(column))
#     plt.legend()
#     plt.show()





###############################
#now start doing modeling and prediction



X_train=all_df[:Y_train.size].copy(deep=True)
X_test=all_df[Y_train.size:].copy(deep=True)
print(X_train.shape, Y_train.shape, X_test.shape)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# machine learning
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def Search_para( model, params, metric, X_data, Y_data, n_fold=5):
    searcher = GridSearchCV(model, params, cv=n_fold, scoring = metric, verbose=1,return_train_score=True,refit=True)
    searcher.fit(X_data, Y_data)
    result_meta=pd.DataFrame(searcher.cv_results_)
    best_params = searcher.best_params_
    best_score = searcher.best_score_

    return (best_params, best_score, result_meta,searcher.best_estimator_)


def Search_Compare(model_list, model_collection, params_collection, metric, X_data, Y_data):
    # store the scores for model comparison
    scores_list = []
    best_params_list = []
    best_estimators=[]
    for model_name, model, params in zip(model_list, model_collection, params_collection):
        res_tup =  Search_para(model, params, metric=metric, X_data=X_data, Y_data=Y_data)
        best_params_list.append(res_tup[0])
        scores_list.append(res_tup[1])
        try:
            score=res_tup[2]['rank_test_score']
            a=np.array(score).reshape((4,7))
            print(a)
        except:
            pass
        print('all scores for this clf: ', res_tup[2][['params','mean_test_score','rank_test_score']].to_string())
        print("model: " + model_name)
        print("Best parameter: {}\n Score: {:5f}".format(res_tup[0], res_tup[1]))
        best_estimators.append(res_tup[3])

    d = {'model': model_list, 'scores': scores_list, 'param': best_params_list}
    res_df = pd.DataFrame(data=d)
    print(res_df.sort_values(by=['scores'], ascending=False).to_string())
    return best_estimators

def add_model(model_names, models, parameters,
              model_name,model, parameter):
    model_names.append(model_name)
    models.append(model)
    parameters.append(parameter)
    return True

model_names=[]
models=[]
parameters=[]

add_model(model_names, models, parameters,
          'Ridge Regression',RidgeClassifier(), {'alpha':[0,0.03,0.05,0.04,0.06,0.07,0.08,0.09,0.092,1,2,4,6]})


add_model(model_names, models, parameters,
          'Logistic Regression', LogisticRegression(max_iter=1000),
          {'penalty':['l1','l2'],'C':[16,17,20,25,28,30,32,34,40,50,60]})
# #
# #
add_model(model_names, models, parameters,
          'SVM (non-linear)', SVC(max_iter=30000,kernel='poly'),
          {'C':[0.05,0.4,0.5,0.6,0.7,1],'degree':[2,3,4,5],'coef0':[0,1,2,3,4]})
          # {'C':[0.5,1, 2, 3,4,5,6,7],'kernel':['linear','poly','rbf','sigmoid']})
#
#best param: C=7.7~8.1
add_model(model_names, models, parameters,
          'SVM (linear)', LinearSVC(max_iter=10000),{'C':[7,7.2,7.4,7.6,7.9,8.1,8.3]})

add_model(model_names, models, parameters,
          'KNN', KNeighborsClassifier(),
          {'n_neighbors':[3,4,5,6,7,8,9,10],'weights':['uniform','distance']})

add_model(model_names, models, parameters,
          'Random Forest', RandomForestClassifier(),
          {'n_estimators':[3,7,8,9,10,11,12,15,30,48,49,51,60],'max_features':[4,5,6,7,8,10,12,15],'min_samples_leaf':[1,2,3,4]})

#n_estimator=490
#max_depth=3
#min_child_weight=4
#gamma=0.5
#tree-related parameters:
#colsample_bytree=0.9, subsample=0.9
#reg_alpha=0,reg_lambda=0.08
cv_params = {'learning_rate': [0.1,0.105,0.11, 0.115,0.120]}
add_model(model_names, models, parameters,
          'XGBoost',
          xgb.XGBClassifier(
              learning_rate =0.1,
              n_estimators=490,
              objective= 'binary:logistic',
              scale_pos_weight=1,
              max_depth=3,
              min_child_weight=4,
              gamma=0.5,
              subsample=0.9,
              colsample_bytree=0.9,
              reg_alpha=0,reg_lambda=0.08,
              seed=20),

          cv_params)
# #
#
best_estimators=Search_Compare(model_list=model_names, model_collection=models, params_collection=parameters,
              metric='accuracy',X_data=X_train, Y_data=Y_train)

# n_neighbors = [4,5,6,7,8,9,10,11,12,14]
# algorithm = ['auto']
# weights = ['uniform', 'distance']
# leaf_size = list(range(1,50,5))
# hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size,
#                'n_neighbors': n_neighbors}
# gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,
#                 cv=5, scoring = "accuracy")
# gd.fit(X_train,Y_train)
# print(gd.best_score_)
# print(gd.best_estimator_)
##########
#output result
# survival_prediction=best_estimators[0].predict(X_test)
# submission=pd.DataFrame({'PassengerId': test_passenger_id,'Survived':survival_prediction})
# submission.to_csv('../output/submission(rf).csv', index=False)