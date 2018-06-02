# Chapter 8 Lab: Decision Trees

# Fitting Classification Trees
library(tree)
library(ISLR)

attach(Carseats)
High=ifelse(Sales<=8,"No","Yes")
Carseats=data.frame(Carseats,High)
tree.carseats=tree(High~.-Sales,Carseats)

summary(tree.carseats)
## Q: what is deviance? A: -2loglik with minimum = 0
plot(tree.carseats)
text(tree.carseats,pretty=0)
## Branches that lead to terminal nodes are indicated using asterisks.
tree.carseats

set.seed(2)
train=sample(1:nrow(Carseats), 200)
Carseats.test=Carseats[-train,]
High.test=High[-train]
tree.carseats=tree(High~.-Sales,Carseats,subset=train)
## Let's training error and test error. Overfitting occurs.
tree.pred.test=predict(tree.carseats,Carseats.test,type="class")
table(tree.pred.test,High.test)
(86+57)/200
tree.pred.train = predict(tree.carseats, Carseats[train,], type='class')
table(tree.pred.train, High[train])
(114+68)/200
## Cross-validation and pruning tree.
set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)

names(cv.carseats)
cv.carseats
## size is # terminal nodes and k is the regularization meta param
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")

## Note here tree.carseats is fitted by the whole training set.
prune.carseats=prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(94+60)/200
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats)
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
(86+62)/200

# Fitting Regression Trees

library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty=0)
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston,pretty=0)
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

# Bagging and Random Forests

library(randomForest)
set.seed(1)
## Note mtry is num of vars sampled as canditate at each split
## importance is a boolean arg, which indicate whether importance of predictors be assessed
## ntree is number of trees. Default = 500
## When mtry = p, then do bagging!
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)
set.seed(1)

rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
## the total decrease in node impurities from splitting on the variable, averaged over all trees. 
## For classification, the node impurity is measured by the Gini index. 
## For regression, it is measured by residual sum of squares.
importance(rf.boston)
varImpPlot(rf.boston)

# Boosting
library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.boston)
## Paritial Dependece Plot: f(x) = E_y(f(x,y)). NOT:E(f(x,y)|x)
par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")

yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
