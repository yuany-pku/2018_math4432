#Math 4432 mini project 2 (chow wing ho 20279607)
rm(list = ls())

Train = read.csv('train.csv')
Test = read.csv('test.csv')

#create saleprice variable in test data
Test$SalePrice = NA
combine = rbind(Train,Test)

dim(combine)
str(combine)

missval = sapply(combine, function(x)  sum(is.na(x)) )
missval

#miscFeature,Alley,PoolQc,FireplaceQu  and Fench are missing too much
#remove them
Drop =  names(combine) %in% c("PoolQC","MiscFeature","Alley","Fence","FireplaceQu")
combine = combine[!Drop]
combine
#fill up the missing value
combine$Utilities = NULL
combine[["MasVnrType"]][is.na(combine[["MasVnrType"]])] = "None"
combine$LotFrontage[is.na(combine$LotFrontage)] = median(combine$LotFrontage, na.rm = T)
combine[["MasVnrArea"]][is.na(combine[["MasVnrArea"]])] = 0
combine[["MSZoning"]][is.na(combine[["MSZoning"]])] = levels(combine[["MSZoning"]])[which.max(table(combine[["MSZoning"]]))]
combine[["Exterior1st"]][is.na(combine[["Exterior1st"]])] = levels(combine[["Exterior1st"]])[which.max(table(combine[["Exterior1st"]]))]
combine[["Exterior2nd"]][is.na(combine[["Exterior2nd"]])] = levels(combine[["Exterior2nd"]])[which.max(table(combine[["Exterior2nd"]]))]
combine[["KitchenQual"]][is.na(combine[["KitchenQual"]])] = levels(combine[["KitchenQual"]])[which.max(table(combine[["KitchenQual"]]))]
combine[["Electrical"]][is.na(combine[["Electrical"]])] = levels(combine[["Electrical"]])[which.max(table(combine[["Electrical"]]))]
combine[["SaleType"]][is.na(combine[["SaleType"]])] = levels(combine[["SaleType"]])[which.max(table(combine[["SaleType"]]))]
combine[["BsmtQual"]][is.na(combine[["BsmtQual"]])] = levels(combine[["BsmtQual"]])[which.max(table(combine[["BsmtQual"]]))]
combine[["BsmtExposure"]][is.na(combine[["BsmtExposure"]])] = levels(combine[["BsmtExposure"]])[which.max(table(combine[["BsmtExposure"]]))]
combine[["BsmtFinType2"]][is.na(combine[["BsmtFinType2"]])] = levels(combine[["BsmtFinType2"]])[which.max(table(combine[["BsmtFinType2"]]))]
combine[["BsmtCond"]][is.na(combine[["BsmtCond"]])] = levels(combine[["BsmtCond"]])[which.max(table(combine[["BsmtCond"]]))]
combine[["BsmtFinType1"]][is.na(combine[["BsmtFinType1"]])] = levels(combine[["BsmtFinType1"]])[which.max(table(combine[["BsmtFinType1"]]))]
combine[["GarageType"]][is.na(combine[["GarageType"]])] = levels(combine[["GarageType"]])[which.max(table(combine[["GarageType"]]))]
combine[["GarageQual"]][is.na(combine[["GarageQual"]])] = levels(combine[["GarageQual"]])[which.max(table(combine[["GarageQual"]]))]
combine[["GarageCond"]][is.na(combine[["GarageCond"]])] = levels(combine[["GarageCond"]])[which.max(table(combine[["GarageCond"]]))]
combine[["GarageFinish"]][is.na(combine[["GarageFinish"]])] = levels(combine[["GarageFinish"]])[which.max(table(combine[["GarageFinish"]]))]
combine[["GarageCars"]][is.na(combine[["GarageCars"]])] = 0
combine[["GarageArea"]][is.na(combine[["GarageArea"]])] = 0
combine[["BsmtFullBath"]][is.na(combine[["BsmtFullBath"]])] = 0
combine[["BsmtHalfBath"]][is.na(combine[["BsmtHalfBath"]])] = 0
combine[["BsmtFinSF1"]][is.na(combine[["BsmtFinSF1"]])] = 0
combine[["BsmtFinSF2"]][is.na(combine[["BsmtFinSF2"]])] = 0
combine[["BsmtUnfSF"]][is.na(combine[["BsmtUnfSF"]])] = 0
combine[["TotalBsmtSF"]][is.na(combine[["TotalBsmtSF"]])] = 0
combine[["Functional"]][is.na(combine[["Functional"]])] = levels(combine[["Functional"]])[which.max(table(combine[["Functional"]]))]
combine$GarageYrBlt[is.na(combine$GarageYrBlt)] = median(combine$GarageYrBlt, na.rm = T)
missval = sapply(combine, function(x)  sum(is.na(x)) )
missval

Tr = combine[!is.na(combine$SalePrice)==TRUE, ]
Te = combine[!is.na(combine$SalePrice)==FALSE, ]


attach(combine)

lm1 = lm(SalePrice ~ LotArea + Neighborhood + Condition1+Condition2+ BldgType + HouseStyle + YearBuilt + YearRemodAdd + OverallQual + OverallCond +MoSold +YrSold,Tr) 
lm2 = lm(SalePrice~HouseStyle)
lm3 = lm(SalePrice~LotArea)
lm4 = lm(SalePrice~Neighborhood)
lm5 = lm(SalePrice~Condition1)
lm6 = lm(SalePrice~Condition2)
lm7 = lm(SalePrice~BldgType)
lm8 = lm(SalePrice~YearBuilt)
lm9 = lm(SalePrice~YearRemodAdd)
lm10 = lm(SalePrice~OverallQual)
lm11 = lm(SalePrice~MoSold)
lm13 = lm(SalePrice~OverallCond)
lm12 = lm(SalePrice~YrSold)

summary(lm1)
summary(lm2)
summary(lm3)
summary(lm4)
summary(lm5)
summary(lm6)
summary(lm7)
summary(lm8)
summary(lm9)
summary(lm10)
summary(lm11)
summary(lm12)
summary(lm13)

#remove YrSold, MoSold
lm14 = lm(SalePrice ~ LotArea + Neighborhood + Condition1+Condition2+ BldgType + HouseStyle + YearBuilt + YearRemodAdd + OverallQual + OverallCond) 
summary(lm14)

#remove OverallCond
lm15 = lm(SalePrice ~ LotArea + Neighborhood + Condition1 + Condition2 
          + BldgType + HouseStyle + YearBuilt + YearRemodAdd + OverallQual,Tr) 
summary(lm15)
par(mfrow=c(2,2))
plot(lm15)

lmpredict = predict(lm(SalePrice ~ LotArea + Neighborhood + Condition1 + Condition2 
           + BldgType + HouseStyle + YearBuilt + YearRemodAdd + OverallQual),Te)
summary(lmpredict)
par(mfrow=c(1,1))
plot(lmpredict,main = "Sales price prediction" )

Te$SalePrice = lmpredict
write.csv(Te, file = "Te.csv")


install.packages("glmnet")
library(glmnet)

LASSOformula = as.formula( log(SalePrice)~ .-Id )

x = model.matrix(LASSOformula, Tr)
y = log(Tr$SalePrice)

set.seed(1234)
lmlasso = cv.glmnet(x, y, alpha=1)
plot(lmlasso)

Te$SalePrice = 1
Te_x = model.matrix(LASSOformula, Te)

lmpred = predict(lmlasso, newx = Te_x, s = "lambda.min")
plot(exp(lmpred), main = "sales price by LASSO")
modprice = data.frame(Id = Te$Id, SalePrice = exp(lmpred))
write.csv(res, file = "predictmod.csv") 
