rm(list=ls())
sp=read.csv("sp.csv")
sum(is.na(sp))
sum(is.na(sp$slowWaveSleep))
install.packages("mice")
require(mice)
mice.sp <- mice(sp,
                 m = 1,           # ????????????????????????????????????
                 maxit = 50,      # max iteration
                 method = "cart", # ??????CART?????????,?????????????????????
                 seed = 188)      # set.seed(),????????????????????????
sp 
complete(mice.sp,1) #data
sp1 = complete(mice.sp,1)
head(sp1)
attach(sp1)
sp10 = rep(0, length(sp1$sleep))
sp10[sp1$sleep > median(sp1$sleep)] = 1
sp1 = data.frame(sp1, sp10)
sp1




attach(sp1)
m1= lm(sp10~slowWaveSleep)
m2= lm(sp10~slowWaveSleep+dreamSleep)
m3= lm(sp10~slowWaveSleep+dreamSleep+brain)
m4= lm(sp10~slowWaveSleep+dreamSleep+brain+body)
m5= lm(sp10~slowWaveSleep+dreamSleep+brain+body+life)
m6= lm(sp10~slowWaveSleep+dreamSleep+brain+body+life+gestation)
m7= lm(sp10~slowWaveSleep+dreamSleep+brain+body+life+gestation+predation)
m8= lm(sp10~slowWaveSleep+dreamSleep+brain+body+life+gestation+predation+danger)
m9= lm(sp10~slowWaveSleep+dreamSleep+brain+body+life+gestation+predation+danger+sleepExposure)
install.packages("stargazer")
library(stargazer)
stargazer(M1,M2,M3,M4,M5,M6,M7,M8,M9,M10
          ,type="html",out="tab.html")

summary(m9)


sp1$predation = as.numeric(sp1$predation)
sp1$sleepExposure = as.numeric(sp1$sleepExposure)
sp1$danger = as.numeric(sp1$danger)
plot(sp1$sp10,sp1$slowWaveSleep)
pairs(~sp1$sp10+sp1$predation+sp1$sleepExposure+sp1$danger,data = sp1)
cor(sp1$sp10,sp1$slowWaveSleep)
cor(sp1$sp10,sp1$gestation)
cor(sp1$sp10,sp1$sleepExposure)
cor(sp1$sp10,sp1$body)
cor(sp1$sp10,sp1$dreamSleep)
cor(sp1$sp10,sp1$brain)
cor(sp1$sp10,sp1$life)
cor(sp1$sp10,sp1$predation)
cor(sp1$sp10,sp1$danger)
attach(sp1)



#abc = lm(sp1$sleep ~ sp1$dreamSleep+sp1$slowWaveSleep+sp1$body+sp1$brain+sp1$life
 #             +sp1$gestation+sp1$predation+sp1$sleepExposure+sp1$danger, 
  #            data = sp1)
#summary(abc)


M1 = glm(sp1$sp10 ~ sp1$dreamSleep
              +sp1$slowWaveSleep
              +sp1$predation
              +sp1$danger+sp1$sleepExposure, 
              data = sp1,family = "binomial")
summary(M1)



sq_danger = sp1$danger ^ 2
sq_pred = sp1$predation ^ 2
sq_se = sp1$sleepExposure^2
sp2 = data.frame(sp1$sp10, sq_pred, sq_danger)
M2= glm(sp1$sp10 ~ sp1$slowWaveSleep + sp1$danger
                   + sp1$predation + sp1$dreamSleep+sq_se, 
                   data = sp1,family = "binomial")
summary(M2)

M3 = glm(sp1$sp10 ~ sp1$slowWaveSleep + sq_danger
         + sp1$predation + sp1$dreamSleep+sp1$sleepExposure, 
         data = sp1,family = "binomial")
summary(M3)
plot(M3)
M4= glm(sp1$sp10 ~ sp1$slowWaveSleep + sp1$danger
        + sq_pred + sp1$dreamSleep, 
        data = sp1,family = "binomial")

summary(M4)


pred_danger = sq_pred*sq_danger
sp1 = data.frame(sp1, pred_danger)
M5 = glm(sp1$sp10 ~ sp1$slowWaveSleep + sp1$dreamSleep 
               + pred_danger, data = sp1,family = "binomial")
summary(M5)
#The p-value of term relating to danger is still high.

sq_slowWaveSleep = sp1$slowWaveSleep ^ 2
M6 = glm(sp1$sp10 ~ sq_slowWaveSleep + sp1$dreamSleep + sp1$danger + sp1$predation, 
               data = sp1, family = "binomial")
summary(M6)


sq_dreamSleep = sp1$dreamSleep ^ 2
M7 = glm(sp1$sp10 ~ sp1$slowWaveSleep + sq_dreamSleep + sp1$danger + sp1$predation, 
         data = sp1, family = "binomial")
summary(M7)

slowwavesleep_dreamSleep = sq_dreamSleep * sq_slowWaveSleep
M8 = glm(sp1$sp10 ~ slowwavesleep_dreamSleep + sp1$danger + sp1$predation, 
         data = sp1, family = "binomial")
summary(M8)

M9 = glm(sp1$sp10 ~sp1$slowWaveSleep +sp1$dreamSleep, data = sp1, family = "binomial")
summary(M9)

M10= glm(sp1$sp10 ~ slowwavesleep_dreamSleep, 
         data = sp1,family = "binomial")
summary(M10)
par(mfrow=c(2,2))
plot(M10)


install.packages("boot")
library(boot)
cv.glm(sp1, M1)$delta[1]
cv.glm(sp1, M2)$delta[1]
cv.glm(sp1, M3)$delta[1]
cv.glm(sp1, M4)$delta[1]
cv.glm(sp1, M5)$delta[1]
cv.glm(sp1, M6)$delta[1]
cv.glm(sp1, M7)$delta[1]
cv.glm(sp1, M8)$delta[1]
cv.glm(sp1, M9)$delta[1]
cv.glm(sp1,M10)$delta[1]
cv.glm(sp1,M10)

install.packages("DAAG")
library(DAAG)
attach(sp1)
abc=lm(sp1$sp10 ~ slowwavesleep_dreamSleep, 
       data = sp1)
summary(abc)
cv.lm(sp1$sp10 ~ slowwavesleep_dreamSleep,data = sp1)


boot.fn = function(data, index){
  M10=glm(sp1$sp10 ~ slowwavesleep_dreamSleep, 
          data = sp1, family = "binomial", subset = index)
  return(coef(M10))
}
boot(sp1, boot.fn, 1000)

coef(M10)
summary(M10)
