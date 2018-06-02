library(ElemStatLearn)
library(splines)
library(glmnet)
data("phoneme")

# 695 * 256
aa_ = phoneme[phoneme$g == 'aa', 1:256]
# 1022 * 256
ao_ = phoneme[phoneme$g == 'ao', 1:256]
data_ = rbind(aa_,ao_)
rownames(data_) = 1:(695+1022)
# 1-aa and 0-ao
data_$label = c(rep(1,695), rep(0,1022))

set.seed(10)
train_idx = sample(1:(695+1022), 1000)
train_ = data_[train_idx,]
test_ = data_[setdiff(1:(695+1022), train_idx),]

plot(1:256, train_[12,1:256], type = 'n')
lines(1:256, train_[12,1:256], lty=1, col='red')
lines(1:256, train_[1000, 1:256], lty= 2, col='blue')
dev.off()

# Natural Cubic Splines
# df = Interiorknots + 2boundaryknots + (3deg+intercept) - 4
ns_theta = ns(1:256, df=18, intercept = F) # 256 * df(basis funcs without intercept)
print(attr(ns_theta,'knots')) # 10 interior knots
print(attr(ns_theta,"Boundary.knots")) # 2 boundary knots
ns_tr_ = data.frame(as.matrix(train_[,1:256]) %*% ns_theta)
ns_ts_ = data.frame(as.matrix(test_[,1:256]) %*% ns_theta)
ns_tr_$label = train_$label
ns_fit = glm(label~., data=ns_tr_, family = 'binomial')
ns_pred = predict(ns_fit, newdata = ns_ts_, type='response')
ns_acc = sum((ns_pred > .5) == test_$label) / dim(test_)[1]
print(ns_acc)
ns_coef = coef(ns_fit)
temp = predict(ns_theta, seq(1,256,.5))

## Logistic Regression
lg_fit = glm(label~., data = train_, family = 'binomial')
lg_pred = predict(lg_fit, newdata = test_, type = 'response')
lg_acc = sum((lg_pred > .5) == test_$label) / dim(test_)[1]
lg_acc
lg_coef = coef(lg_fit)

## Logistic Regression with L_1 penalty
plg_fit = cv.glmnet(x=as.matrix(train_[,1:256]), y=train_[,257], 
                    family='binomial', type.measure = 'class',
                    nfolds = 5)
plg_pred = predict(plg_fit,newx=as.matrix(test_[,1:256]),
                   s=plg_fit$lambda.min,type='class')
plg_acc = sum((as.numeric(plg_pred) > .5) == test_$label) / dim(test_)[1]
plg_acc
plg_coef = coef(plg_fit)

plot(1:256, lg_coef[2:257], type='l')
lines(1:256, plg_coef[2:257],col='red')
lines(seq(1,256,.5), temp %*% ns_coef[2:19], col='blue')
