#Read and check rawdata
traindata = read.csv('C:/Users/Xingbo SHANG/Desktop/HKUST courses/MATH4432/Project2/train.csv')
testdata = read.csv('C:/Users/Xingbo SHANG/Desktop/HKUST courses/MATH4432/Project2/test.csv')
#Change id from an observable to row name
rownames(traindata) =traindata$Id
rownames(testdata) = testdata$Id
traindata = traindata[,-1]
testdata = testdata[,-1]
fix(traindata)
fix(testdata)

################################################################################
################################################################################
#                       Section for NA value handling                          #
################################################################################
################################################################################

#Check NA values in the dataset
num_NA_train = rep(0,ncol(traindata))
num_NA_test = rep(0,ncol(testdata))
for(n in 1:ncol(testdata))
{
  num_NA_train[n] = sum(is.na(traindata[,n]))
  num_NA_test[n] = sum(is.na(testdata[,n]))
}
par(mfrow = c(1,2))
par(las = 2)
par(mar=c(5,8,4,2))
barplot(num_NA_train[num_NA_train!=0],
        names.arg = colnames(traindata)[num_NA_train!=0],
        main = 'NA value summary of train data',
        horiz = TRUE,cex.names = 0.8)
barplot(num_NA_test[num_NA_test!=0],
        names.arg = colnames(testdata)[num_NA_test!=0],
        main = 'NA value summary of test data',
        horiz = TRUE,cex.names = 0.8)
par(las = 0)
par(mar = c(5,4,4,2))
par(mfrow = c(1,1))
#Handle NA values in the data
#Note, NA values in LotFrontage is not imputed by the function NA_value_handling because
#NA values in LotFrontage is more complicated

source('C:/Users/Xingbo SHANG/Downloads/NA_value_handling.R')
train.modified = NA_value_handling(traindata)
test.modified = NA_value_handling(testdata)
#Ensure that every factor in test.modified and train.modified has the same level.
#Otherwise, there will be problems in predict.
for (n in 1:ncol(test.modified))
{
  if(is.factor(test.modified[,n]))
  {
    levels(test.modified[,n]) = union(levels(test.modified[,n]),levels(train.modified[,n]))
    levels(train.modified[,n]) = union(levels(test.modified[,n]),levels(train.modified[,n]))
  }
}
#Explore the relationship between LotFrontage and other predictors
for(n in 1:(ncol(train.modified)-1))
{
  directory = '/Users/collinzhangkao/Desktop/Year 4 Spring/MATH 4432/Project2/plots/'
  jpeg(paste('LotFrontage against ',colnames(train.modified)[n],'.jpeg',sep = ''),width = 800, height = 800)
  plot(train.modified$LotFrontage~train.modified[,n],col = 'red',varwidth = TRUE,
       xlab = colnames(train.modified)[n],ylab = 'LotFrontage',
       main = paste('LotFrontage against',colnames(train.modified)[n],sep = ' '))
  dev.off()
}

#Treatment of NA values in LotFrontage
library(randomForest)
data_for_imputation = rbind(train.modified[,-81],test.modified)
training_for_imputation = data_for_imputation[data_for_imputation$LotFrontageAva == 'Yes',]
#Use validation set approach to demosntrate that randomforest is significantly better than median approximation.
set.seed(1)
training.index = sample(1:nrow(training_for_imputation),0.75*nrow(training_for_imputation))
training = training_for_imputation[training.index,]
validation = training_for_imputation[-training.index,]
rf.LotFrontage = randomForest(LotFrontage~.,data = training)
rf.imputation = predict(rf.LotFrontage, newdata = validation)
rf.imputation.error = sqrt(mean((rf.imputation-validation$LotFrontage)^2))
median.error = sqrt(mean((median(validation$LotFrontage)-validation$LotFrontage)^2))
rf.imputation.error
median.error
#Use randomForest to estimate NA values in LotFrontage
rf.LotFrontage = randomForest(LotFrontage~.,
                              data = data_for_imputation[data_for_imputation$LotFrontageAva == 'Yes',],
                              importance = TRUE)
varImpPlot(rf.LotFrontage, main = 'Importance of differnt predictors for LotFrontage')
rf.imputation = predict(rf.LotFrontage, 
                        newdata = train.modified[train.modified$LotFrontageAva == 'No',])
train.final = train.modified
train.final$LotFrontage[train.modified$LotFrontageAva == 'No'] = round(rf.imputation)

rf.imputation = predict(rf.LotFrontage, 
                        newdata = test.modified[test.modified$LotFrontageAva == 'No',])
test.final = test.modified
test.final$LotFrontage[test.modified$LotFrontageAva == 'No'] = round(rf.imputation)
#Check NA values in modified data to see whether all NA values have been handled
num_NA_train = rep(0,ncol(traindata))
num_NA_test = rep(0,ncol(testdata))
for(n in 1:ncol(test.final))
{
  num_NA_train[n] = sum(is.na(train.final[,n]))
  num_NA_test[n] = sum(is.na(test.final[,n]))
}
num_NA_train
colnames(train.final)[num_NA_train!=0]
num_NA_test
colnames(test.final)[num_NA_test!=0]

################################################################################
################################################################################
#                End of section for NA value handling                          #
################################################################################
################################################################################

#Log transformation of response variable 
train.final$SalePrice = log(train.final$SalePrice)
#Split the dataset into training and validation dataset
#This validation set is mainly for us to estimate the performance of each model and decide which model to submit
#for fianl evaluation.
set.seed(1)
training.index = sample(1:nrow(train.final),0.75*nrow(train.final))
training = train.final[training.index,]
validation = train.final[-training.index,]

################################################################################
################################################################################
#                Section for Ridge Regression and Lasso                        #
################################################################################
################################################################################
library(glmnet)
training.x = model.matrix(SalePrice~.,training)[,-1]
training.y = training$SalePrice
validation.x = model.matrix(SalePrice~.,validation)[,-1]
#Apply lasso regression
lasso.cv = cv.glmnet(training.x,training.y,alpha = 1)
lasso.bestlam =lasso.cv$lambda.min
lasso.price = glmnet(training.x,training.y,alpha = 1)
lasso.predict = predict(lasso.price,s = lasso.bestlam, newx = validation.x)
lasso.error = sqrt(mean((lasso.predict-validation$SalePrice)^2))
lasso.error
lasso.coef = predict(lasso.price, s=lasso.bestlam, type = 'coefficients')[1:80,]
#Display non-zero coefficients
lasso.coef[lasso.coef!=0]
#Calculate number of non-zero coefficients.
#Minus one because intercept is not a predictor
length(lasso.coef[lasso.coef!=0])-1
#Apply ridge regression
ridge.cv = cv.glmnet(training.x,training.y,alpha = 0)
ridge.bestlam = ridge.cv$lambda.min
ridge.price = glmnet(training.x,training.y,alpha = 0)
ridge.predict = predict(ridge.price,s=ridge.bestlam,newx=validation.x)
ridge.error = sqrt(mean((ridge.predict-validation$SalePrice)^2))
ridge.error
ridge.coef = predict(ridge.price, s = ridge.bestlam, type = 'coefficients')[1:80,]
ridge.coef


################################################################################
################################################################################
#        Section for modelling using tree, random forest and boosting          #
################################################################################
################################################################################

#Apply tree method to understand major features of consideration for price.
library(tree)
tree.price = tree(SalePrice~.,data = training)
summary(tree.price)
plot(tree.price)
text(tree.price,pretty = 0)
set.seed(2)
cv.price=cv.tree(tree.price, K = 30)
plot(cv.price$size,cv.price$dev,type = 'b',xlab = 'Number of nodes',ylab = '')
pruned.price = prune.tree(tree.price,best = 8)
plot(pruned.price)
text(pruned.price,pretty = 0)
pruned.predict = predict(pruned.price,newdata = validation)
error.tree = sqrt(mean((pruned.predict-validation$SalePrice)^2))
error.tree
#Apply random forests (including bagging) to predict price.
library(randomForest)
set.seed(2)
#Bagging
bag.price = randomForest(SalePrice~.,data = training,mtry = 80,importance = TRUE)
plot(bag.price)
varImpPlot(bag.price)
bag.predict = predict(bag.price,newdata = validation)
bag.error = sqrt(mean((bag.predict-validation$SalePrice)^2))
bag.error
#Randomforest
rf.price = randomForest(SalePrice~.,data = training,importance = TRUE)
plot(rf.price)
rf.predict = predict(rf.price,newdata = validation)
rf.error = sqrt(mean((rf.predict-validation$SalePrice)^2))
rf.error
varImpPlot(rf.price,main = 'Impotrance of predictors calculated by random forest')
#Apply boosting to predict price.
library(gbm)
boost.error = rep(0,10)
#Optimize the interaction depth of boosting, one can also use this loop to adjust n.trees  
for (m in 1:10)
{
  set.seed(2)
  boost.price = gbm(SalePrice~.,data = training,distribution = 'gaussian',
                    n.trees = 20000,interaction.depth = m)
  boost.predict = predict(boost.price, newdata = validation,n.trees = 20000)
  boost.error[m] = sqrt(mean((boost.predict-validation$SalePrice)^2))
  print(paste('Finish n =',as.character(n),', m = ',as.character(m)))
}

plot(boost.error~c(1:10),type = 'b',
     main = 'Boosting with different interaction depth',
     xlab = 'Interaction depth',ylab = 'Error'
     )

#The best model we decide is n.trees = 20000 and interaction.depth = 4.
boost.price = gbm(SalePrice~., data = train.final, distribution = 'gaussian',
                  n.trees = 20000, interaction.depth = 4)
summary(boost.price)
test.predict = predict(boost.price, newdata = test.final,n.trees = 20000)
result = data.frame(rownames(testdata),exp(test.predict))
colnames(result) = c('Id','SalePrice')
write.csv(result, file = 'C:/Users/Xingbo SHANG/Desktop/HKUST courses/MATH4432/Project2/boosting_submission.csv')

################################################################################
################################################################################
#     End of section for modelling using tree, random forest and boosting      #
################################################################################
################################################################################