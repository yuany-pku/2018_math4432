#Create col names
time = c(rep(0, 81))
for (i in c(1:81))
{
  if (i%%3 == 1)
  {
    time[i] = paste("time", format(i%/%3), ".count", sep = "", collapse = "")
  }
  
  if (i%%3 == 2)
  {
    time[i] = paste("time", format(i%/%3), ".mean", sep = "", collapse = "")
  }
  
  if (i%%3 == 0)
  {
    time[i] = paste("time", format(i%/%3 - 1), ".sd", sep = "", collapse = "")
  }
}
time = c("machine", "date", time, "status")
train = read.csv("C:/Users/Anson/Documents/R/Training freq 1D, OW 2, PW 1.csv", header = FALSE, col.names = time, na.strings = "", sep=",")
test = read.csv("C:/Users/Anson/Documents/R/Training freq 1D, OW 4, PW 1.csv", header = FALSE, col.names = time, na.strings = "", sep=",")


#Remove rows 1,2 and 3
train= train[-c(1,2,3), ]

#Set the status as 1 for True, 0 for False
sum(is.na(train$status))
#There are 14 na which is a small amount compared to
#the sample size. So we omit the missing data
train = train[!is.na(train$status), ]
train$status = as.character(train$status)
train$status[train$status == "True"] = 1
train$status[train$status == "False"] = 0
train$status = as.factor(train$status)

#Fill in the space with 0
for(i in 3:83){
  train[,i]=as.numeric(as.character(train[,i]))
  test[,i]=as.numeric(as.character(test[,i]))
}
for(i in 3:83){
  if(grepl(".count", colnames(train[i]))){
    for(j in 1:nrow(train)){
      if(is.na(train[j,i])){
        train[j,i] = 0
      }
    }
  }
  if(grepl(".mean", colnames(train[i]))){
    for(j in 1:nrow(train)){
      if(is.na(train[j,i])){
        train[j,i] = 0
      }
    }
  }
  if(grepl(".sd", colnames(train[i]))){
    for(j in 1:nrow(train)){
      if(is.na(train[j,i])){
        train[j,i] = 0
      }
    }
  }
}

#Convert the date into categorical factor
train[ ,2] = as.Date(train[ ,2], "%Y-%m-%d")

title1 = c(seq(1505, 1512, by = 1), seq(1601, 1612, by = 1), 
           seq(1701, 1712, by = 1), seq(1801, 1802, by = 1))
title1 = as.character(title1)
status1.train = data.frame(matrix(0, nrow = nrow(train), ncol = 34))
colnames(status1.train) = title1
title1 = as.numeric(title1)

for (i in c(1:nrow(train)))
{
  for (j in c(1:34))
  {
    year = paste("20", format(title1[j]%/%100), sep = "", collapse = "")
    if (title1[j] %% 100 > 9) {
      month = paste(format(title1[j] %% 100))
    } else {
      month = paste("0", format(title1[j] %% 100), sep = "", collapse = "")
    }
    start1 = paste(year, month, "01", sep = "-")
    if ((year == "16") && (month == "02")) {
      end1 = paste(year, month, "29", sep = "-")
    } else if ((month == "01")|| (month == "03")||(month == "05")|| (month == "07")||(month == "08")|| (month == "10")||(month == "12")) {
      end1 = paste(year, month, "31", sep = "-")
    } else if(month == "02") {
      end1 = paste(year, month, "28", sep = "-")
    } else {
      end1 = paste(year, month, "30", sep = "-")}
    start1 = as.Date(start1)
    end1 = as.Date(end1)
    date1 = seq(start1, end1, by = 1)
    for(k in date1)
    {
      if (train[i, 2] == k)
        status1.train[i, j] = 1
    }
  }
}

#Set the last group as the reference
status1.train = status1.train[ , -34]

#combine the column in status 1 into groups
#Set the last group as the reference
status2.train = data.frame(matrix(0,nrow = nrow(train), ncol = 6))
title2 = c(1501, 1507, 1601, 1607, 1701, 1707)
title2 = as.character(title2)
colnames(status1.train) = title2
status2.train[ ,1] = status1.train[ ,1] + status1.train[ ,2]
status2.train[ ,2] = status1.train[ ,3] + status1.train[ ,4] + status1.train[ ,5] + status1.train[ ,6] + status1.train[ ,7] + status1.train[ ,8] 
status2.train[ ,3] = status1.train[ ,9] + status1.train[ ,10] + status1.train[ ,11] + status1.train[ ,12] + status1.train[ ,13] + status1.train[ ,14] 
status2.train[ ,4] = status1.train[ ,15] + status1.train[ ,16] + status1.train[ ,17] + status1.train[ ,18] + status1.train[ ,19] + status1.train[ ,20] 
status2.train[ ,5] = status1.train[ ,21] + status1.train[ ,22] + status1.train[ ,23] + status1.train[ ,24] + status1.train[ ,25] + status1.train[ ,26] 
status2.train[ ,6] = status1.train[ ,27] + status1.train[ ,28] + status1.train[ ,29] + status1.train[ ,30] + status1.train[ ,31] + status1.train[ ,32] 

#Remove the date 
train = train[ ,-c(1,2)]
train1 = cbind(train, status1.train)
train2 = cbind(train, status2.train)
train1 = data.frame(train1)
train2 = data.frame(train2)

#library
library(tree)
library(randomForest)
library(MASS)
library(class)
library(e1071)
library(pROC)
library(ROSE)


#Seperate the set intto a training set and a test set.
set.seed(1)
index = sample(dim(train1)[1], dim(train1)[1] * 0.7)
train.train = train1[index, ]
train.test = train1[-index, ] 

#Classification tree
tree.train = tree(status ~ ., data = train.train)
summary(tree.train)
#time14.count is an important factor

tree.pred = predict(tree.train, train.test, type = "class")
table(tree.pred, train.test$status)
mean(tree.pred != train.test$status)


rf.train = randomForest(status ~ ., data = train.train, 
                        mrty = sqrt(115), ntree = 1000)
summary(rf.train)
tree.pred = predict(rf.train, train.test, type = "class")
table(tree.pred, train.test$status)
mean(tree.pred != train.test$status)
importance(rf.train, asscending = TRUE)


#logistic regression
#Add the factors that has a significant importance to status
logis.train1 = glm(status ~ time6.count + time6.mean + time6.sd + time8.count + time8.mean + time8.sd +
                     time14.count + time14.mean + time14.sd + time15.count + time15.mean + time15.sd  +
                     time17.count + time17.mean + time17.sd , data = train.train, family = "binomial")
probs1 = predict(logis.train1, train.test, type = "response")
pred1 = rep(0, dim(train.test)[1])
pred1[probs1 > 0.5] = 1
table(pred1, train.test$status)
mean(pred1 != train.test$status)
#The error rate increases a bit

#Add the factors of time14
logis.train2 = glm(status ~ time14.count + time14.mean + time14.sd, data = train.train, family = "binomial")
probs2 = predict(logis.train2, train.test, type = "response")
pred2 = rep(0, dim(train.test)[1])
pred2[probs2 > 0.5] = 1
table(pred2, train.test$status)
mean(pred2 != train.test$status)
#The error rate is the same as that of logis.train1

#Just add the factor of time14.count
logis.train3 = glm(status ~ time14.count, data = train.train, family = "binomial")
probs3 = predict(logis.train3, train.test, type = "response")
pred3 = rep(0, dim(train.test)[1])
pred3[probs3 > 0.5] = 1
table(pred3, train.test$status)
mean(pred3 != train.test$status)

lda.train = lda(status ~ time14.count + time14.mean + time14.sd, data = train.train)
lda.pred = predict(lda.train, newdata = train.test, type = "response")
table(lda.pred$class, train.test$status)
mean(lda.pred$class != train.test$status)
#The error rate is the same

qda.train = qda(status ~ time14.count + time14.mean + time14.sd, data = train.train)
qda.pred = predict(qda.train, newdata = train.test, type = "response")
table(qda.pred$class, train.test$status)
mean(qda.pred$class != train.test$status)
#The error rate increases.

train.X1 = cbind(train.train$time14.count, train.train$time14.mean, train.train$time14.sd)
test.X1 = cbind(train.test$time14.count, train.test$time14.mean, train.test$time14.sd)
train.Y = cbind(train.train$status)
knn.pred1 = knn(train.X1, test.X1, train.Y, k = 1)
table(knn.pred1, train.test$status)
#The error rate increases a lot

knn.pred2 = knn(train.X1, test.X1, train.Y, k = 5)
table(knn.pred2, train.test$status)
#The error rate decreases a bit

knn.pred3 = knn(train.X1, test.X1, train.Y, k = 10)
table(knn.pred3, train.test$status)
#The error rate decrease a bit

svm.train1 = svm(status ~ time14.count + time14.mean + time14.sd, data = train.train, kernel = "linear", 
                 cost = 0.01)
svm.pred1 = predict(svm.train1, newdata = train.test, type = "response")
table(svm.pred1, train.test$status)
mean(svm.pred1 != train.test$status)

svm.train2 = svm(status ~ time14.count + time14.mean + time14.sd, data = train.train, kernel = "radial", 
                 cost = 0.01, gamma = 1)
svm.pred2 = predict(svm.train2, newdata = train.test, type = "response")
table(svm.pred2, train.test$status)
mean(svm.pred2 != train.test$status)

svm.train3 = svm(status ~ time14.count + time14.mean + time14.sd, data = train.train, kernel = "polynomial", 
                 cost = 0.01, degree = 2)
svm.pred3 = predict(svm.train3, newdata = train.test, type = "response")
table(svm.pred3, train.test$status)
mean(svm.pred3 != train.test$status)

#bootstrap
boot.fn = function(data, index){
  fit.glm = glm(sleep01 ~ slowWaveSleep + dreamSleep + danger, data = data,
                family = "binomial", subset = index)
  return(coef(fit.glm))
}

#test
test = read.csv("C:/Users/Anson/Documents/R/Training freq 1D, OW 4, PW 1.csv", header = FALSE, col.names = time, na.strings = "", sep=",")
test= train[-c(1,2,3), ]

test = test[!is.na(train$status), ]
test$status = as.character(test$status)
test$status[test$status == "True"] = 1
test$status[test$status == "False"] = 0
test$status = as.factor(test$status)

for(i in 3:83){
  test[,i]=as.numeric(as.character(test[,i]))
}



for(i in 3:83){
  if(grepl(".count", colnames(test[i]))){
    for(j in 1:nrow(test)){
      if(is.na(test[j,i])){
        test[j,i] = 0
      }
    }
  }
  if(grepl(".mean", colnames(test[i]))){
    for(j in 1:nrow(test)){
      if(is.na(test[j,i])){
        test[j,i] = 0
      }
    }
  }
  if(grepl(".sd", colnames(test[i]))){
    for(j in 1:nrow(test)){
      if(is.na(test[j,i])){
        test[j,i] = 0
      }
    }
  }
}

title1 = as.character(title1)
status1.test = data.frame(matrix(0, nrow = nrow(test), ncol = 34))
colnames(status1.test) = title1
title1 = as.numeric(title1)
for (i in c(1:nrow(test)))
{
  for (j in c(1:34))
  {
    year = paste("20", format(title1[j]%/%100), sep = "", collapse = "")
    if (title1[j] %% 100 > 9) {
      month = paste(format(title1[j] %% 100))
    } else {
      month = paste("0", format(title1[j] %% 100), sep = "", collapse = "")
    }
    start1 = paste(year, month, "01", sep = "-")
    if ((year == "16") && (month == "02")) {
      end1 = paste(year, month, "29", sep = "-")
    } else if ((month == "01")|| (month == "03")||(month == "05")|| (month == "07")||(month == "08")|| (month == "10")||(month == "12")) {
      end1 = paste(year, month, "31", sep = "-")
    } else if(month == "02") {
      end1 = paste(year, month, "28", sep = "-")
    } else {
      end1 = paste(year, month, "30", sep = "-")}
    start1 = as.Date(start1)
    end1 = as.Date(end1)
    date1 = seq(start1, end1, by = 1)
    for(k in date1)
    {
      if (test[i, 2] == k)
        status1.test[i, j] = 1
    }
  }
}
test = test[ ,-c(1,2)]
test1 = cbind(test, status1.test)
test1 = data.frame(test1)

train.X1 = cbind(train.train$time14.count, train.train$time14.mean, train.train$time14.sd)
test.X1 = cbind(test1$time14.count, test1$time14.mean, test1$time14.sd)
train.Y = cbind(train.train$status)
knn.pred1 = knn(train.X1, test.X1, train.Y, k = 1)
table(knn.pred1, test1$status)
roc.curve(test1$status, knn.pred1)
