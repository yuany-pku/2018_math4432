# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:39:08 2018

@author: Junrong
"""

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#load train-x and train-y
filename = 'train.txt'

num_train = 7291
num_feature = 256

train = np.zeros(shape = (num_train,num_feature+1))
train_x = np.zeros(shape = (num_train,num_feature))
train_y = np.zeros(num_train)

train = np.loadtxt(filename)

for i in range(0,num_train):
    train_y[i] = train[i][0]
    for j in range(1,num_feature+1):
        train_x[i][j-1] = train[i][j]
print("finish loading training data")

#load test-x and test-y
filename = 'test.txt'

num_test = 2007
num_feature = 256

test = np.zeros(shape = (num_test,num_feature+1))
test_x = np.zeros(shape = (num_test,num_feature))
test_y = np.zeros(num_test)

test = np.loadtxt(filename)

for i in range(0,num_test):
    test_y[i] = test[i][0]
    for j in range(1,num_feature+1):
        test_x[i][j-1] = test[i][j]

print("finish loading testing data")

#######################################
############   XGBOOST   ##############
#######################################
"""


print("start to build the model")
model = XGBClassifier()
model.fit(train_x, train_y)

print("model ready")

print(model)

pred_y = np.zeros(num_test)

# make predictions for test data
pred_y = model.predict(test_x)
predictions = [round(value) for value in pred_y]

#check error rate
error_test = 0.0

for i in range (0,num_test):
    if(pred_y[i] != test_y[i]):
        error_test += 1
    
error_test = error_test / num_test

print("XGB: test error")
print(error_test)

"""

#######################################
######   K Nearest Neighbours   #######
#######################################
#this is the K nearest neighbours algorithm
#the hyperparameter is K here
#we can modify that simply by modifying the knn_k value

'''
print("start modeling on KNN")

from sklearn.neighbors import KNeighborsClassifier

index_x = np.zeros(50)
train_error_list = np.zeros(50)
test_error_list = np.zeros(50)

for index_k in range (1,51):
    knn_k = index_k

    print("working on train x")
    x, y = train_x, train_y
    clf = KNeighborsClassifier(n_neighbors=knn_k)
    clf.fit(x, y)
    print("finished")

 
    print(clf)


    print("predict on train x")
    predict_result_train = clf.predict(train_x)
    print("finished")

    print("predict on test x")
    predict_result_test = clf.predict(test_x)
    print("finished")

    #error rate calculation
    error_train = 0.0
    error_test = 0.0

    for i in range (0,num_train):
        if(predict_result_train[i] != train_y[i]):
            error_train += 1
    
    error_train = error_train / num_train


    for i in range (0,num_test):
        if(predict_result_test[i] != test_y[i]):
            error_test += 1
    
    error_train = error_train / num_train
    train_error_list[index_k - 1] = error_train
    
    error_test = error_test / num_test
    test_error_list[index_k - 1] = error_test

    index_x[index_k - 1] = index_k
    #print("K Nearest Neighbours: train error")
    #print(error_train)

    #print("K Nearest Neighbours: test error")
    #print(error_test)
plt.plot(index_x,train_error_list,index_x,test_error_list)
plt.title("Error rate versus index k")
plt.show()
'''
    
#######################################
#######   Linear Regression   #########
#######################################

'''

print("start modeling on Linear Regression")
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the testing set
predict_result_test = regr.predict(test_x)
print(regr)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, predict_result_test))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, predict_result_test))
'''

#######################################
###########      SGD      #############
#######################################
#this is the SGD built in sklearn library
'''

print("start on SGD")
from sklearn import linear_model

print("working on train x")
clf = linear_model.SGDClassifier()
clf.fit(train_x, train_y)
print("finished")

print(clf)


print("predict on train x")
predict_result_train = clf.predict(train_x)
print("finished")

print("predict on test x")
predict_result_dev = clf.predict(test_x)
print("finished")

#error rate calculation
error_train = 0.0
error_test = 0.0

for i in range (0,num_train):
    if(predict_result_train[i] != train_y[i]):
        error_train += 1
    
error_train = error_train / num_train


for i in range (0,num_test):
    if(predict_result_test[i] != test_y[i]):
        error_test += 1
    
error_train = error_train / num_train
error_test = error_test / num_test


print("SGD: train error")
print(error_train)

print("SGD: test error")
print(error_test)
'''

#######################################
#######      Naive Bayes      #########
#######################################
#this is the naive bayes algorithm built in sklearn and it uses the gaussian naive bayes


'''
print("start on Naive Bayes")
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

print("working on train x")
gnb = gnb.fit(train_x, train_y)
print("finished")

print(gnb)


print("predict on train x")
predict_result_train = gnb.predict(train_x)
print("finished")

print("predict on test x")
predict_result_dev = gnb.predict(test_x)
print("finished")

#error rate calculation
error_train = 0.0
error_test = 0.0

for i in range (0,num_train):
    if(predict_result_train[i] != train_y[i]):
        error_train += 1
    
error_train = error_train / num_train


for i in range (0,num_test):
    if(predict_result_test[i] != test_y[i]):
        error_test += 1
    
error_train = error_train / num_train
error_test = error_test / num_test


print("Naive Bayes: train error")
print(error_train)

print("Naive Bayes: test error")
print(error_test)
'''

#######################################
########    Neural Network    #########
#######################################
#this is the part for neural network
#the parameters can be tuned inside the MLPClassifier
#hidden_layer_sizes is the number of layers and the number of neurons at each layer
#also max_iter is the number of iteraions in this neural network
#this is a algorithm with some uncertainty and it may perform with a large difference even though
#the parameters settings are the same
#for the best performance on the kaggle, I used this method and the settings are as bellow
#hidden_layer_sizes=(200,200), max_iter=3000

'''
print("start to make neural network")
#regularize data
print("data regularization")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x)
StandardScaler(copy=True, with_mean=True, with_std=True)
train = scaler.transform(train_x)


#train the model
print("start to train the model")
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(200,200,50,50,50,50,50,50,50),max_iter=3000)
mlp.fit(train_x,train_y)
print("mlp parameters: ")
print(mlp)

print("start to predict on train")
train_pred = mlp.predict(train_x)
print("finished")

print("start to predict on test")
test_pred = mlp.predict(test_x)
print("finished")



#error rate calculation
error_train = 0.0
error_test = 0.0

for i in range (0,num_train):
    if(predict_result_train[i] != train_y[i]):
        error_train += 1
    
error_train = error_train / num_train


for i in range (0,num_test):
    if(predict_result_test[i] != test_y[i]):
        error_test += 1
    
error_train = error_train / num_train
error_test = error_test / num_test


print("Neural Network: train error")
print(error_train)

print("Neural Network: test error")
print(error_test)

'''

#######################################
##############    LDA    ##############
#######################################
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = train_x, train_y

x_cor = np.arange(100) + 1
y_cor = np.zeros(100)

for n in range (1,101):
    lda = LinearDiscriminantAnalysis(n_components = n)
    lda.fit(X, y)
    #y_hat = lda.predict(test_x)
    #y_cor[n-1] = np.sum(y_hat == test_y) / y.shape[0]
    
    y_hat = lda.predict(test_x)
    y_cor[n-1] = np.sum(y_hat == test_y) / test_y.shape[0]
    
    #print ("the sklearn lda accuracy in n=", n)  
    #print (np.sum(y_hat == test_y) / y.shape[0] )
    #print (lda.coef_)
    #write to test_yhat.txt
    #np.savetxt("output1.txt",lda.coef_)
    #print (lda.means_)
    #print (len(lda.score(X, y)))
    #print (lda.scalings_)
    #print (lda.priors_)

    #X_new = lda.transform(X)
    #plt.scatter(X_new, y,marker='o',c=y)
    #plt.show()

plt.plot(x_cor,y_cor)
'''


#######################################
##############    QDA    ##############
#######################################
'''
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X, y = train_x, train_y

x_cor = (np.arange(101) + 1) * 0.01

y_cor = np.zeros(101)


para_list = np.zeros(101)
for i in range(0,101):
    para_list[i] = 0.01 * i
  
for p in range(0,101):    
    qda = QuadraticDiscriminantAnalysis(reg_param = para_list[p])
    qda.fit(X, y)
    y_hat = qda.predict(test_x)
    y_cor[p] = np.sum(y_hat == test_y) / test_y.shape[0]
    
    #y_hat = qda.predict(X)
    #y_cor[p] = np.sum(y_hat == y) / y.shape[0]
    
    #print ("the sklearn qda accuracy : = ", p)
    #print (np.sum(y_hat == test_y) / y.shape[0])


plt.plot(x_cor,y_cor)
'''


#######################################
############    PCA & LDA   ###########
#######################################

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X, y = test_x, test_y
x_cor = np.arange(100) + 1
y_cor = np.zeros(100)

for n in range(0,100):
    pca = PCA(n_components=n+1)
    pca.fit(test_x)
    #print (pca.explained_variance_ratio_)
    #print (pca.explained_variance_)
    X = pca.fit_transform(test_x)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    #y_hat = lda.predict(test_x)
    #y_cor[n-1] = np.sum(y_hat == test_y) / y.shape[0]
    
    y_hat = lda.predict(X)
    y_cor[n] = np.sum(y_hat == test_y) / test_y.shape[0]
    
    #print ("the sklearn lda accuracy in n=", n)  
    #print (np.sum(y_hat == test_y) / y.shape[0] )
    #print (lda.coef_)
    #write to test_yhat.txt
    #np.savetxt("output1.txt",lda.coef_)
    #print (lda.means_)
    #print (len(lda.score(X, y)))
    #print (lda.scalings_)
    #print (lda.priors_)

    #X_new = lda.transform(X)
    #plt.scatter(X_new, y,marker='o',c=y)
    #plt.show()

plt.plot(x_cor,y_cor)

