from __future__ import print_function, division
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.ensemble import AdaBoostClassifier as adaboost
from sklearn.ensemble import RandomForestClassifier as randomforest
from sklearn.svm import SVC


def xgb_classifier(eta, rounds, x_train, y_train, x_test, y_test):
    temp_x_train = pd.DataFrame(x_train)
    temp_x_test = pd.DataFrame(x_test)
    print("Start to train xgb")
    feature = temp_x_train.columns
    temp_x_train['label'] = y_train
    temp_x_test['label'] = y_test
    dtrain = xgb.DMatrix(temp_x_train[feature], temp_x_train.label)
    dtest = xgb.DMatrix(temp_x_test[feature], temp_x_test.label)
    params = {
        'objective':'binary:logistic',
        'eta': eta,
        'eval_metric': 'error',
        'seed': 100,
        'silent': 1
        }
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    num_rounds = rounds
    eval_result = {}
    model = xgb.train(params = params, 
                      dtrain = dtrain, 
                      num_boost_round = num_rounds, 
                      evals=watchlist,
                      evals_result= eval_result, 
                      verbose_eval = True,
                      )
    y_predict = pd.Series(model.predict(dtest)).map(lambda x: 1 if x>0.5 else 0)
    acc = sum(y_predict==y_test['0'])/(y_test.shape[0])
    print('XGB test accuracy is: %f'%acc)
    return model
    
# adaboost
def ada_classifier(x_train, y_train, x_test, y_test):
    print("Start to train adaboost")
    model = adaboost()
    model.fit(x_train, y_train['0'])
    print("Adaboost accuracy is: %f"%model.score(x_test, y_test))
    return model

# randomforest
def randomforest_classifier(m_depth, n_esti ,x_train, y_train, x_test, y_test):
    print("Start to train random forest")
    clf = randomforest(n_estimators=n_esti, max_depth=m_depth)
    clf.fit(x_train, y_train['0'])
    score = clf.score(x_test, y_test)
    print("Random Forest accuracy is: %f"%score )
    return clf, score


def svm_classifier(kernel, x_train, y_train, x_test, y_test):
    print("Start to train SVM")
    svm = SVC(kernel = kernel, probability = False)
    svm.fit(x_train, y_train['0'])
    print("SVM accuracy is: %f"%svm.score(x_test, y_test))
    return svm




if __name__ == '__main__':
    x_train = pd.read_csv('train_feature.csv', header=0)
    y_train = pd.read_csv('y_train.csv', header=0)
    x_test = pd.read_csv('test_feature.csv', header=0)
    y_test = pd.read_csv('y_test.csv', header=0)

    svm_classifier(x_train, y_train, x_test, y_test)
