#This is the source code for mini project 1 from Xingbo SHANG and Kao ZHANG

#Read and check the csv file.
rawdata = read.csv('https://raw.githubusercontent.com/yuany-pku/data/master/wells.csv',
                 header = TRUE,
                 sep = ',')
fix(rawdata)

#Some modifications to the rawdata for convenience
rawdata$switch=as.factor(rawdata$switch)
rawdata$unsafe=as.factor(rawdata$unsafe)

#Part A 
#Preliminary exploration of the relationship between the respone 'Switch' and other variables.

par(mfrow=c(2,2))
for(n in c(2,4,7,8))
{
  plot(rawdata[,n]~rawdata$switch, col = 'red',varwidth = TRUE,
       xlab = 'Switch',ylab = colnames(rawdata)[n],
       main = paste(colnames(rawdata)[n],' vs switch'))
}
par(mfrow=c(1,1))

#Unsafe is a dummy variable and is not so good to do boxplot.
#Use table insted.
table(rawdata$switch,rawdata$unsafe)

#Conclusion for part A:
#arsenic seems to be the most significant preidictor for switch.
#distance and unsafe also seems to be a potential predictor.
#community and education does not have significant correlation with switch, so they will not be further explored as predictors.


#Part B
#A general exploration on LDA, Logistic_Regresion, QDA and KNN's performance on the dataset. 
#With arsenic as the single predictor, since it is both intuitively and stastically related to switch. 
#General idea: randomly split into the data into 100 pairs of training and validation sets.
#100 error rates will be calculated for each four models (LDA, QDA, Logistic_Regresion and KNN).
#Compare the error rates using side by side box plot to pick the best model.

#Before fitting models, set up a dataframe called error_rate to store error rates for all models.
#400 = 4*100. Four different models in total and 100 error rartes to store for each model.
error_rate = data.frame(rep(0,400),rep(0,400))
colnames(error_rate) = c('Error_rate','Model_type')

library(MASS)
library(class)
for(n in 1:100)
{
  #Prepare traning and validation dataset for model fitting.
  set.seed(n)
  train.index = sample(1:nrow(rawdata),size = 0.75*nrow(rawdata))
  training = rawdata[train.index,]
  validation = rawdata[-train.index,]
  
  #Logistic, LDA, QDA and KNN with arsenic as the single predictor
  #Logistic regression with arsenic as the single predictor
  glm1.fit = glm(switch~arsenic, data = training, family = 'binomial')
  glm1.predict = predict(glm1.fit,validation,type = 'response')
  glm1.results = rep(TRUE,nrow(validation))
  glm1.results[glm1.predict<0.5] = FALSE
  error_rate$Error_rate[n] = mean(glm1.results!=validation$switch)
  error_rate$Model_type[n] = 'Logistic_Regression'
  
  #LDA with arsenic as the single predictor
  lda1.fit = lda(switch~arsenic, data = training, family = 'binomial')
  lda1.predict = predict(lda1.fit,validation)
  lda1.results=lda1.predict$class
  error_rate$Error_rate[100+n] = mean(lda1.results!=validation$switch)
  error_rate$Model_type[100+n] = 'LDA'
  
  #QDA with aresnic as the single predictor
  qda1.fit = qda(switch~arsenic, data = training, family = 'binomial')
  qda1.predict = predict(lda1.fit,validation)
  qda1.results=qda1.predict$class
  error_rate$Error_rate[200+n] = mean(qda1.results!=validation$switch)
  error_rate$Model_type[200+n] = 'QDA'
  
  #KNN with aresnic as the single predictor
  set.seed(201)
  KNN_errors = rep(0,10)
  for(m in 1:10)
  {
    knn.predict = knn(data.frame(training$arsenic),data.frame(validation$arsenic),training$switch,k=m)
    KNN_errors[m] = mean(knn.predict!=validation$switch)
  }
  error_rate$Error_rate[300+n] = min(KNN_errors)
  error_rate$Model_type[300+n] = 'KNN'
}

plot(error_rate$Error_rate~as.factor(error_rate$Model_type),col = 'red',
     main = 'Error rates of four different models',
     xlab = 'Model types',ylab = 'Error rate')

#Conclusion for part B:
#LDA and QDA have the highest error rate,but why? => Part C
#KNN performs the best=>there can be some non-linear relationship between switch and arsenic.
#This means that it is possible to improve the logistic regression model by including non-linearity=>Part D
#Even if KNN performs the best, we prefer improving logistic regression instead of directly using KNN because:
#a) As a non-parametric method, it is difficult to interpret the model.
#b) It suffers from curse of dimensionality when more predictors are included.

#Part C
#Explain why LDA and QDA has significantly lower performance

#TRUE_arsenic stores all the arsenic values for wells with switch==TRUE.
#FALSE_arsenic stores all the arsenic values for wells with switch==FALSE.
TRUE_arsenic = rawdata$arsenic[rawdata$switch==TRUE]
FALSE_arsenic = rawdata$arsenic[rawdata$switch==FALSE]

TRUE_arsenic_standardized = (TRUE_arsenic-mean(TRUE_arsenic))/sd(TRUE_arsenic)
FALSE_arsenic_standardized = (TRUE_arsenic-mean(FALSE_arsenic))/sd(FALSE_arsenic)

#Plot QQ plot and histogram to show that arsenic values are not normally distributed.
par(mfrow = c(2,2))
qqnorm(TRUE_arsenic_standardized, 
       main = 'QQplot of standardized arsenic level of switched wells')
qqline(TRUE_arsenic_standardized)
hist(TRUE_arsenic_standardized,breaks = 100, 
     main = 'Histogram of standardized arsenic level of switched wells',
     xlab = 'Starndardized arsenic level')

qqnorm(FALSE_arsenic_standardized, 
       main = 'QQplot of standardized arsenic level of unswitched wells')
qqline(FALSE_arsenic_standardized)
hist(FALSE_arsenic_standardized,breaks = 100, 
     main = 'Histogram of standardized arsenic level of unswitched wells',
     xlab = 'Standardized arsenic level')
par(mfrow = c(1,1))
#Conclusion for Part C:
#It is not surprising that LDA and QDA perform quite bad on the dataset.
#arsenic values significantly deviate from normal distribution in both true and false class.
#This violates the Gaussian assumption of LDA and QDA, but logistic regression and KNN do not have this assumption.
#This non-Gaussian situation is quite common in pollution level because:
#a)Unpolluted areas all tend to cluster around certain value close to zero
#b)For polluted areas, the level can go up to very high.
#The final result is an assymmetric distribution, which maybe better modeled by gamma distribution. 

#Part D logistic regression with non-linear trend.
#Seven different transformations are tried: polynomial from order 1 to 5, log transformation and square root transformation.

#Before fitting models, set up a dataframe called error_rate_nonlinear to store error rates for all models.
error_rate_nonlinear_glm = data.frame(rep(0,700),rep(0,700))
colnames(error_rate_nonlinear_glm) = c('Error_rate','Model_type')
for(n in 1:100)
{
  #Prepare traning and validation dataset for model fitting.
  set.seed(n)
  train.index = sample(1:nrow(rawdata),size = 0.75*nrow(rawdata))
  training = rawdata[train.index,]
  validation = rawdata[-train.index,]
  
  for(m in 1:7)
  {
    #m==6 corresponds to log transformation
    if(m==6)
    {
      glm.nonlinear.fit = glm(switch~sqrt(arsenic), data = training, family = 'binomial')
      error_rate_nonlinear_glm$Model_type[100*(m-1)+n] = 'Square_root'
    }
    #m==7 corresponds to square root transformation
    else if(m==7)
    {
      glm.nonlinear.fit = glm(switch~log(arsenic), data = training, family = 'binomial')
      error_rate_nonlinear_glm$Model_type[100*(m-1)+n] = 'Log'
    }
    #m==1~5 corresponds to polynomial transformation of order 1~5
    else
    {
      glm.nonlinear.fit = glm(switch~poly(arsenic,m), data = training, family = 'binomial')
      error_rate_nonlinear_glm$Model_type[100*(m-1)+n] = m
    }
    glm.nonlinear.predict = predict(glm.nonlinear.fit,validation,type = 'response')
    glm.nonlinear.results = rep(TRUE,nrow(validation))
    glm.nonlinear.results[glm.nonlinear.predict<0.5] = FALSE
    error_rate_nonlinear_glm$Error_rate[100*(m-1)+n] = mean(glm.nonlinear.results!=validation$switch)
  }
}

plot(error_rate_nonlinear_glm$Error_rate~as.factor(error_rate_nonlinear_glm$Model_type),
     col = 'red',
     main = 'Error rates of logistic regression with different transformations',
     xlab = 'Type of transfomation applied to arsenic',
     ylab = 'Error rate')

#Conclusion for Part D
#Including non-linear trend into logistic regression indeed improves the performance of the model.
#Log transformation of arsenic is chosen to be the most faborable model because:
#a) It has the lowest error rate (statistically not different from polynomial with order 3,4,5)
#b) It only requires two parameters to be inferred when compared with polunomial with order 3,4,5.
#To conclude, it gets the lowest bias with the lowest possible variance.

#Get an estimate of coefficients and confusion matruix of logistic regression with log transformation
set.seed(101)
train.index = sample(1:nrow(rawdata),size = 0.75*nrow(rawdata))
training = rawdata[train.index,]
validation = rawdata[-train.index,]

glm.log.fit = glm(switch~log(arsenic),data = training, family = 'binomial')
summary(glm.log.fit)
glm.log.predict = predict(glm.log.fit,validation,type = 'response')
glm.log.results = rep(TRUE,nrow(validation))
glm.log.results[glm.log.predict<0.5] = FALSE

cmatrix = table(glm.log.results,validation$switch)
cmatrix

#Part E
#Try including other predictors (distance and unsafe) into logistic regression with log transformation
set.seed(101)
train.index = sample(1:nrow(rawdata),size = 0.75*nrow(rawdata))
training = rawdata[train.index,]
validation = rawdata[-train.index,]

glm.log.fit = glm(switch~log(arsenic)+distance,data = training, family = 'binomial')
summary(glm.log.fit)

glm.log.fit = glm(switch~log(arsenic)+unsafe,data = training, family = 'binomial')
summary(glm.log.fit)


glm.log.fit = glm(switch~log(arsenic)+distance+unsafe,data = training, family = 'binomial')
summary(glm.log.fit)
#Conslusion from part E:
#All parameters have significant coefficients, but does this mean they should be included?=>Part F

#Part F
#Check whether including extra parameters (distance and unsafe) indeed has practical meaning (aka, lower error rates).

#Before fitting models, set up a dataframe called error_rate_nonlinear to store error rates for all models.
error_rate_predictors_glm = data.frame(rep(0,400),rep(0,400))
colnames(error_rate_predictors_glm) = c('Error_rate','Model_type')
for(n in 1:100)
{
  #Prepare traning and validation dataset for model fitting.
  set.seed(n)
  train.index = sample(1:nrow(rawdata),size = 0.75*nrow(rawdata))
  training = rawdata[train.index,]
  validation = rawdata[-train.index,]
  
  for(m in 1:4)
  {
    if(m==1)
    {
      glm.fit = glm(switch~log(arsenic), data = training, family = 'binomial')
      error_rate_predictors_glm$Model_type[100*(m-1)+n] = 'A'
    }
    else if(m==2)
    {
      glm.fit = glm(switch~log(arsenic)+distance, data = training, family = 'binomial')
      error_rate_predictors_glm$Model_type[100*(m-1)+n] = 'A+D'
    }
    else if (m==3)
    {
      glm.fit = glm(switch~log(arsenic)+unsafe, data = training, family = 'binomial')
      error_rate_predictors_glm$Model_type[100*(m-1)+n] = 'A+U'
    }
    else if (m==4)
    {
      glm.fit = glm(switch~log(arsenic)+unsafe+distance, data = training, family = 'binomial')
      error_rate_predictors_glm$Model_type[100*(m-1)+n] = 'A+U+D'
    }
    glm.predict = predict(glm.fit,validation,type = 'response')
    glm.results = rep(TRUE,nrow(validation))
    glm.results[glm.predict<0.5] = FALSE
    error_rate_predictors_glm$Error_rate[100*(m-1)+n] = mean( glm.results!=validation$switch)
  }
}

plot(error_rate_predictors_glm$Error_rate~as.factor(error_rate_predictors_glm$Model_type),
     col = 'red',
     main = 'Error rates of logistic regression with different predictors included',
     xlab = 'Type of predictors included',
     ylab = 'Error rate')

#Reason why unsafe does not improve the performance of the model
plot(rawdata$arsenic~rawdata$unsafe,col = 'red',varwidth = TRUE,
     main = 'Arsenic level vs unsafe',
     xlab = 'Unsafe',ylab = 'Arsenic level')

#Conclusion from Part F:
#Including distance into the model indeed improves the performance.
#unsafe does not help with improving the performance because it is co-linear with arsenic.
#You can predict whether a well is unsafe or not simply based on arsenic.
#unsafe only shares the prediction power from arsenic, it is not necessary to include unsafe into the model.

#Part G
#Evaluation of the final model to be used, aka, logistic regression with switch~log(arsenic)+distance
#Simply a code to generate ROC curve.
set.seed(1)
train.index = sample(1:nrow(rawdata),size = 0.75*nrow(rawdata))
training = rawdata[train.index,]
validation = rawdata[-train.index,]

threshold = seq(0,1,0.01)
tp = rep(0,101)
fp = rep(0,101)

for(n in 1:101)
{
  glm.fit=glm(switch~log(arsenic)+distance,data = training, family = 'binomial')
  glm.predict = predict(glm.fit,validation,type = 'response')
  glm.results = rep(FALSE,nrow(validation))
  glm.results[glm.predict>threshold[n]]=TRUE
  tp[n] = sum(glm.results==TRUE&validation$switch==TRUE)/sum(validation$switch==TRUE)
  fp[n] = sum(glm.results==TRUE&validation$switch!=TRUE)/sum(validation$switch==FALSE)
}

plot(tp~fp,type = 'l',col = 'red',lwd = 2,
     main = 'ROC curve of the final model',
     xlab = 'False positive rate',ylab = 'True positive rate',
     xlim = c(0,1))

abline(a = 0, b = 1, lty = 2)

