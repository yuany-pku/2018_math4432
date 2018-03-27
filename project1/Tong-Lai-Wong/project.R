#fill in na data by median
library(ISLR)
library(MASS)
library(boot)
set.seed(1)
sleep = read.csv("C:/Users/Anson/Documents/R/sleep1.csv")

sleep$slowWaveSleep[is.na(sleep$slowWaveSleep)] = median(sleep$slowWaveSleep[!is.na(sleep$slowWaveSleep)])
sleep$dreamSleep[is.na(sleep$dreamSleep)] = median(sleep$dreamSleep[!is.na(sleep$dreamSleep)])
sleep$sleep[is.na(sleep$sleep)] = median(sleep$sleep[!is.na(sleep$sleep)])
sleep$life[is.na(sleep$life)] = median(sleep$life[!is.na(sleep$life)])
sleep$gestation[is.na(sleep$gestation)] = median(sleep$gestation[!is.na(sleep$gestation)])

sleep01 = rep(0, length(sleep$sleep))
sleep01[sleep$sleep > median(sleep$sleep)] = 1
sleep = data.frame(sleep, sleep01)

par(mfrow=c(2,3))
boxplot(predation ~ sleep01, data = sleep, main = "Sleep01 vs Predation")
boxplot(sleepExposure ~ sleep01, data = sleep, main = "Sleep01 vs SleepExposure")
boxplot(danger ~ sleep01, data = sleep, main = "Sleep01 vs Danger")
plot(sleep$sleep01, sleep$predation, main = "Sleep01 vs Predation")
plot(sleep$sleep01, sleep$sleepExposure, main = "Sleep01 vs SleepExposure")
plot(sleep$sleep01, sleep$danger, main = "Sleep01 vs Danger")
#Danger and predation may have relationship with sleep.

pairs(sleep) 
#slowWaveSleep and dreamSleep may have relationship with sleep01

fit.glm = glm(sleep01 ~ slowWaveSleep + dreamSleep 
              + danger + predation, 
              data = sleep,family = "binomial")
summary(fit.glm)
#The p-values of danger and predation are very high.

sq_danger = sleep$danger ^ 2
sq_pred = sleep$predation ^ 2
sleep = data.frame(sleep, sq_danger, sq_pred)
fit.glm2 = glm(sleep01 ~ slowWaveSleep + dreamSleep 
               + sq_danger + sq_pred, 
               data = sleep,family = "binomial")
summary(fit.glm2)
#The p-values do not improve a lot.

pred_danger = sleep$sq_pred * sleep$sq_danger
sleep = data.frame(sleep, pred_danger)
fit.glm3 = glm(sleep01 ~ slowWaveSleep + dreamSleep 
               + pred_danger , 
               data = sleep,family = "binomial")
summary(fit.glm3)
#The p-value of term relating to danger is still high.

fit.glm4 = glm(sleep01 ~ slowWaveSleep + dreamSleep + danger, 
               data = sleep, family = "binomial")
summary(fit.glm4)

cv.glm(sleep, fit.glm)$delta[1]
cv.glm(sleep, fit.glm2)$delta[1]
cv.glm(sleep, fit.glm3)$delta[1]
cv.glm(sleep, fit.glm4)$delta[1]
#The forth model has the lowest MSE 




boot.fn = function(data, index){
  fit.glm = glm(sleep01 ~ slowWaveSleep + dreamSleep + danger, data = data,
                family = "binomial", subset = index)
  return(coef(fit.glm))
}
boot(sleep, boot.fn, 1000)
#The std. error is so high. Filling NA by the median may not be appropriate.

sleep2 = read.csv("C:/Users/Anson/Documents/R/sleep1.csv")
num_na = sum(is.na(sleep2$slowWaveSleep))
sample_slow = sample(sleep2$slowWaveSleep[!is.na(sleep2$slowWaveSleep)], num_na, replace = TRUE)
sleep2$slowWaveSleep[is.na(sleep2$slowWaveSleep)] = sample_slow

num_na = sum(is.na(sleep2$dreamSleep))
sample_dream = sample(sleep2$dreamSleep[!is.na(sleep2$dreamSleep)], num_na, replace = TRUE)
sleep2$dreamSleep[is.na(sleep2$dreamSleep)] = sample_dream

num_na = sum(is.na(sleep2$sleep))
sample_sleep = sample(sleep2$sleep[!is.na(sleep2$sleep)], num_na, replace = TRUE)
sleep2$sleep[is.na(sleep2$sleep)] = sample_sleep

num_na = sum(is.na(sleep2$life))
sample_life = sample(sleep2$life[!is.na(sleep2$life)], num_na, replace = TRUE)
sleep2$life[is.na(sleep2$life)] = sample_life

num_na = sum(is.na(sleep2$gestation))
sample_gest = sample(sleep2$gestation[!is.na(sleep2$gestation)], num_na, replace = TRUE)
sleep2$gestation[is.na(sleep2$gestation)] = sample_gest

sleep01 = rep(0, length(sleep2$sleep))
sleep01[sleep2$sleep > median(sleep2$sleep)] = 1
sleep2 = data.frame(sleep2, sleep01)

pairs(sleep2)

par(mfrow=c(2,3))
boxplot(predation ~ sleep01, data = sleep2, main = "Sleep01 vs Predation")
boxplot(sleepExposure ~ sleep01, data = sleep2, main = "Sleep01 vs SleepExposure")
boxplot(danger ~ sleep01, data = sleep2, main = "Sleep01 vs Danger")
plot(sleep2$sleep01, sleep2$predation, main = "Sleep01 vs Predation")
plot(sleep2$sleep01, sleep2$sleepExposure, main = "Sleep01 vs SleepExposure")
plot(sleep2$sleep01, sleep2$danger, main = "Sleep01 vs Danger")

new_fit.glm = glm(sleep01 ~ slowWaveSleep + dreamSleep 
                  + predation + danger, 
                  data = sleep2,family = "binomial")
summary(new_fit.glm)
#The p-values of dreamSleep and predation are  high.

sq_danger = sleep2$danger ^ 5
sq_pred = sleep2$predation ^ 2
sleep2 = data.frame(sleep2, sq_pred, sq_danger)
new_fit.glm2 = glm(sleep01 ~ slowWaveSleep + dreamSleep 
               + sq_pred + sq_danger, 
               data = sleep2,family = "binomial")
summary(new_fit.glm2)
#The p-values of predation is still high.

pred_danger = sleep2$sq_pred  * sleep2$sq_danger
sleep2 = data.frame(sleep2, pred_danger)
new_fit.glm3 = glm(sleep01 ~ slowWaveSleep + dreamSleep  
               + pred_danger , 
               data = sleep2,family = "binomial")
summary(new_fit.glm3)
#The interacting term has a high p-value.
#From the above two models, we can conclude that predation may
#not have much relationship with sleep01.

new_fit.glm4 = glm(sleep01 ~ slowWaveSleep + dreamSleep 
                   + sq_danger , 
                   data = sleep2,family = "binomial")
summary(new_fit.glm4)

cv.glm(sleep2, new_fit.glm)$delta[1]
cv.glm(sleep2, new_fit.glm2)$delta[1]
cv.glm(sleep2, new_fit.glm3)$delta[1]
cv.glm(sleep2, new_fit.glm4)$delta[1]


new_boot.fn = function(data, index){
  fit.glm = glm(sleep01 ~ slowWaveSleep + dreamSleep + sq_danger
                , data = data, family = "binomial"
                , subset = index)
  return(coef(fit.glm))
}

boot(sleep2, new_boot.fn, 1000)
#The std. error increases.

#Therefore, fit.glm4 should be the most suitable one.


