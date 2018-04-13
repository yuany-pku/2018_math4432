library(plyr)
library(tree)
library(gbm)
library(glmnet)
library(randomForest)
library(pls)
set.seed(1)
train = read.csv("C:/Users/Anson/Documents/R/train.csv", stringsAsFactors = F)
test = read.csv("C:/Users/Anson/Documents/R/test.csv", stringsAsFactors = F)

#Fill the missing data by median/mode/resamping
getmode <- function(v) 
{
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
Id = test$Id
train$Id = NULL
test$Id = NULL
SalePrice = c(rep(0, dim(test)[1]))
test = data.frame(test, SalePrice)
all = rbind(train, test)
all$MSZoning[is.na(all$MSZoning)] = getmode(all$MSZoning[!is.na(all$MSZoning)])
all$LotFrontage[is.na(all$LotFrontage)] = median(all$LotFrontage[!is.na(all$LotFrontage)])
all$Utilities[is.na(all$Utilities)] = getmode(all$Utilities[!is.na(all$Utilities)])
all$MasVnrType[is.na(all$MasVnrType)] = getmode(all$MasVnrType[!is.na(all$MasVnrType)])
all$KitchenQual[is.na(all$KitchenQual)] = getmode(all$KitchenQual[!is.na(all$KitchenQual)])
all$MasVnrArea[is.na(all$MasVnrArea)] = median(all$MasVnrArea[!is.na(all$MasVnrArea)])
all$BsmtExposure[is.na(all$BsmtExposure)] = getmode(all$BsmtExposure[!is.na(all$BsmtExposure)])
all$Electrical[is.na(all$Electrical)] = getmode(all$Electrical[!is.na(all$Electrical)])
all$Functional[is.na(all$Functional)] = getmode(all$Functional[!is.na(all$Functional)])
all$Exterior1st[is.na(all$Exterior1st)] = getmode(all$Exterior1st[!is.na(all$Exterior1st)])
all$Exterior2nd[is.na(all$Exterior2nd)] = getmode(all$Exterior2nd[!is.na(all$Exterior2nd)])
all$BsmtFinSF1[is.na(all$BsmtFinSF1)] = median(all$BsmtFinSF1[!is.na(all$BsmtFinSF1)])
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] = median(all$BsmtFinSF2[!is.na(all$BsmtFinSF2)])
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] = median(all$BsmtUnfSF[!is.na(all$BsmtUnfSF)])
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] = median(all$TotalBsmtSF[!is.na(all$TotalBsmtSF)])
all$GarageCars[is.na(all$GarageCars)] = median(all$GarageCars[!is.na(all$GarageCars)])
all$GarageArea[is.na(all$GarageArea)] = median(all$GarageArea[!is.na(all$GarageArea)])
all$SaleType[is.na(all$SaleType)] = getmode(all$SaleType[!is.na(all$SaleType)])

#Fill in the NA data with 0 / "none".
all$BsmtFullBath[is.na(all$BsmtFullBath)] = 0
all$BsmtHalfBath[is.na(all$BsmtHalfBath)] = 0
all$GarageYrBlt[is.na(all$GarageYrBlt)] = 0
all$Alley[is.na(all$Alley)] = "None"
all$BsmtQual[is.na(all$BsmtQual)] = "None"
all$BsmtCond[is.na(all$BsmtCond)] = "None"
all$BsmtFinType1[is.na(all$BsmtFinType1)] = "None"
all$BsmtFinType2[is.na(all$BsmtFinType2)] = "None"
all$FireplaceQu[is.na(all$FireplaceQu)] = "None"
all$GarageType[is.na(all$GarageType)] = "None"
all$GarageFinish[is.na(all$GarageFinish)] = "None"
all$GarageQual[is.na(all$GarageQual)] = "None"
all$GarageCond[is.na(all$GarageCond)] = "None"
all$PoolQC[is.na(all$PoolQC)] = "None"
all$Fence[is.na(all$Fence)] = "None"
all$MiscFeature[is.na(all$MiscFeature)] = "None"

#Revalue some categoric variables and change them into factors
all$MSZoning = as.factor(revalue(all$MSZoning, c('A' = 1, 'C' = 2, 'C (all)'= 2, 'FV' = 3, 'I' = 4, 'RH' = 5, 'RL' = 6, 'RP' = 7, 'RM' = 8)))
all$Street = as.factor(revalue(all$Street, c('Grvl' = 1, 'Pave' = 2)))
all$Alley = as.factor(revalue(all$Alley, c('None' = 0, 'Grvl' = 1, 'Pave' = 2)))
all$LotShape = as.factor(revalue(all$LotShape, c('Reg' = 4, 'IR1' = 3,'IR2' = 2, 'IR3' = 1)))
all$LandContour = as.factor(revalue(all$LandContour, c('Lvl' = 1, 'Bnk' = 2,'HLS' = 3, 'Low' = 4)))
all$Utilities = as.factor(revalue(all$Utilities, c('AllPub' = 4, 'NoSewr' = 3,'NoSeWa' = 2, 'ELO' = 1)))
all$LotConfig = as.factor(revalue(all$LotConfig, c('Inside' = 1, 'Corner' = 2,'CulDSac' = 3, 'FR2' = 4, 'FR3' = 5)))
all$LandSlope = as.factor(revalue(all$LandSlope, c('Gtl' = 3, 'Mod' = 2,'Sev' = 1)))
all$Neighborhood = as.factor(revalue(all$Neighborhood, c('Blmngtn' = 1, 'Blueste' = 2,'BrDale' = 3, 'BrkSide' = 4, 'ClearCr' = 5, 'CollgCr' = 6, 'Crawfor' = 7, 'Edwards' = 8,
                                                   'Gilbert' = 9, 'IDOTRR' = 10, 'MeadowV' = 11, 'Mitchel' = 12, 'NAmes' = 13, 'NoRidge' = 14, 'NPkVill' = 15, 'NridgHt' = 16,
                                                   'NWAmes' = 17, 'OldTown' = 18, 'SWISU' = 19, 'Sawyer' = 20, 'SawyerW' = 21, 'Somerst' = 22, 'StoneBr' = 23, 'Timber' = 24, 'Veenker' = 25)))
all$Condition1 = as.factor(revalue(all$Condition1, c('Artery' = 1, 'Feedr' = 2,'Norm' = 3, 'RRNn' = 4, 'RRAn' = 5, 'PosN' = 6, 'PosA' = 7, 'RRNe' = 8, 'RRAe' = 9)))
all$Condition2 = as.factor(revalue(all$Condition2, c('Artery' = 1, 'Feedr' = 2,'Norm' = 3, 'RRNn' = 4, 'RRAn' = 5, 'PosN' = 6, 'PosA' = 7, 'RRNe' = 8, 'RRAe' = 9)))
all$BldgType = as.factor(revalue(all$BldgType, c('1Fam' = 1, '2fmCon' = 2,'Duplex' = 3, 'TwnhsE' = 4, 'Twnhs' = 5)))
all$HouseStyle = as.factor(revalue(all$HouseStyle, c('1Story' = 1, '1.5Fin' = 2,'1.5Unf' = 3, '2Story' = 4, '2.5Fin' = 5, '2.5Unf' = 6, 'SFoyer' = 7, 'SLvl' = 8)))
all$RoofStyle = as.factor(revalue(all$RoofStyle, c('Flat' = 1, 'Gable' = 2,'Gambrel' = 3, 'Hip' = 4, 'Mansard' = 5, 'Shed' = 6)))
all$RoofMatl = as.factor(revalue(all$RoofMatl, c('ClyTile' = 1, 'CompShg' = 2,'Membran' = 3, 'Metal' = 4, 'Roll' = 5, 'Tar&Grv' = 6, 'WdShake' = 7, 'WdShngl' = 8)))
all$Exterior1st = as.factor(revalue(all$Exterior1st, c('AsbShng' = 1, 'AsphShn' = 2,'Brk Cmn' = 3, 'BrkComm' = 3, 'BrkFace' = 4, 'CBlock' = 5, 'CemntBd' = 6, 'HdBoard' = 7, 'ImStucc' = 8,
                                                 'MetalSd' = 9, 'Other' = 10, 'Plywood' = 11, 'PreCast' = 12, 'Stone' = 13, 'Stucco' = 14, 'VinylSd' = 15, 'Wd Sdng' = 16, 'WdShing' = 17)))
all$Exterior2nd = as.factor(revalue(all$Exterior2nd, c('AsbShng' = 1, 'AsphShn' = 2,'Brk Cmn' = 3, 'BrkFace' = 4, 'CBlock' = 5, 'CmentBd' = 6, 'HdBoard' = 7, 'ImStucc' = 8,
                                                 'MetalSd' = 9, 'Other' = 10, 'Plywood' = 11, 'PreCast' = 12, 'Stone' = 13, 'Stucco' = 14, 'VinylSd' = 15, 'Wd Shng' = 16, 'WdShing' = 17)))
all$MasVnrType = as.factor(revalue(all$MasVnrType, c('None' = 0, 'BrkCmn' = 1,'BrkFace' = 2, 'CBlock' = 3, 'Stone' = 4)))
all$ExterQual = as.factor(revalue(all$ExterQual, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1)))
all$ExterCond = as.factor(revalue(all$ExterCond, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1)))
all$Foundation = as.factor(revalue(all$Foundation, c('BrkTil' = 1, 'CBlock' = 2,'PConc' = 3, 'Slab' = 4, 'Stone' = 5, 'Wood' = 6)))
all$BsmtQual = as.factor(revalue(all$BsmtQual, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1, 'None' = 0)))
all$BsmtCond = as.factor(revalue(all$BsmtCond, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1, 'None' = 0)))
all$BsmtExposure = as.factor(revalue(all$BsmtExposure, c('No' = 0, 'Mn' = 1,'Av' = 2, 'Gd' = 3)))
all$BsmtFinType1 = as.factor(revalue(all$BsmtFinType1, c('None' = 0, 'Unf' = 0, 'LwQ' = 1, 'Rec' = 2, 'BLQ' = 3, 'ALQ' = 4, 'GLQ' = 5)))
all$BsmtFinType2 = as.factor(revalue(all$BsmtFinType2, c('None' = 0, 'Unf' = 0, 'LwQ' = 1, 'Rec' = 2, 'BLQ' = 3, 'ALQ' = 4, 'GLQ' = 5)))
all$Heating = as.factor(revalue(all$Heating, c('Floor' = 1, 'GasA' = 2, 'GasW' = 3, 'Grav' = 4, 'OthW' = 5, 'Wall' = 6)))
all$HeatingQC = as.factor(revalue(all$HeatingQC, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1)))
all$CentralAir = as.factor(revalue(all$CentralAir, c('Y' = 1, 'N' = 0)))
all$Electrical = as.factor(revalue(all$Electrical, c('SBrkr' = 1, 'FuseA' = 2, 'FuseF' = 3, 'FuseP' = 4, 'Mix' = 5)))
all$KitchenQual = as.factor(revalue(all$KitchenQual, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1)))
all$Functional = as.factor(revalue(all$Functional, c('Typ' = 1, 'Min1' = 2, 'Min2' = 3, 'Mod' = 4, 'Maj1' = 5, 'Maj2' = 6, 'Sev' = 7, 'Sal' = 8)))
all$FireplaceQu = as.factor(revalue(all$FireplaceQu, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1, 'None' = 0)))
all$GarageType = as.factor(revalue(all$GarageType, c('None' = 0, '2Types' = 1, 'Attchd' = 2, 'Basment' = 3, 'BuiltIn' = 4, 'CarPort' = 5, 'Detchd' = 6)))
all$GarageFinish = as.factor(revalue(all$GarageFinish, c('None' = 0, 'Unf' = 1, 'RFn' = 2, 'Fin' = 3)))
all$GarageQual = as.factor(revalue(all$GarageQual, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1, 'None' = 0)))
all$GarageCond = as.factor(revalue(all$GarageCond, c('Ex' = 5, 'Gd' = 4, 'TA' = 3, 'Fa' = 2, 'Po' = 1, 'None' = 0)))
all$PavedDrive = as.factor(revalue(all$PavedDrive, c('N' = 1, 'P' = 2, 'Y' = 3)))
all$PoolQC = as.factor(revalue(all$PoolQC, c('Ex' = 4, 'Gd' = 3, 'TA' = 2, 'Fa' = 1, 'None' = 0)))
all$Fence = as.factor(revalue(all$Fence, c('GdPrv' = 4, 'MnPrv' = 3, 'GdWo' = 2, 'MnWw' = 1, 'None' = 0)))
all$MiscFeature = as.factor(revalue(all$MiscFeature, c('Elev' = 5, 'Gar2' = 4, 'Shed' = 3, 'TenC' = 2, 'Othr' = 1, 'None' = 0)))
all$SaleType = as.factor(revalue(all$SaleType, c('Oth' = 1, 'ConLD' = 2, 'ConLI' = 3, 'ConLw' = 4, 'Con' = 5, 'COD' = 6, 'New' = 7, 'VWD' = 8, 'CWD' = 9, 'WD' = 10)))
all$SaleCondition = as.factor(revalue(all$SaleCondition, c('Normal' = 6, 'Abnorml' = 5, 'AdjLand' = 4, 'Alloca' = 3, 'Family' = 2, 'Partial' = 1)))

train = all[1:nrow(train), ]
test = all[(nrow(train) + 1):nrow(all), ]

#Construct a regression tree
index = sample(dim(train)[1], dim(train)[1] * 0.7)
train.train = train[index, ]
train.test = train[-index, ]
tree.train = tree(SalePrice ~ ., data = train.train)
summary(tree.train)
pred.tree = predict(tree.train, newdata = train.test)
mean((pred.tree - train.test$SalePrice) ^ 2)
plot(tree.train)
text(tree.train, pretty = 0)


cv.train= cv.tree(tree.train)
plot(cv.train$size, cv.train$dev, type = "b")
tree.min = which.min(cv.train$dev)
points(cv.train$size[tree.min], cv.train$dev[tree.min],
       col = "red", cex = 2, pch = 20)
#The optimal node is 12. So we do not need to prune tree

bag.train = randomForest(SalePrice ~ ., data = train.train, 
                            mtry = 79, ntree = 1000,
                            importance = TRUE)
pred.bag = predict(bag.train, newdata = train.test)
mean((pred.bag - train.test$SalePrice) ^ 2)
#The test error decreases.

rf.train = randomForest(SalePrice ~ ., data = train.train, 
                         mtry = 9, ntree = 1000,
                         importance = TRUE)
pred.rf = predict(rf.train, newdata = train.test)
mean((pred.rf - train.test$SalePrice) ^ 2)
#The test error decreases.

importance(rf.train)
#"GrLivArea", "Neighborhood" and "TotalBsmtSF" are the most important 
#factors according to random forest.

train.mat = model.matrix(SalePrice ~ ., data = train.train)
test.mat = model.matrix(SalePrice ~ ., data = train.test)
grid = 10 ^ seq(10, -2, length = 100)
fit.ridge = glmnet(train.mat, train.train$SalePrice, alpha = 0,
                   lambda = grid, thresh = 1e-12)
cv.ridge = cv.glmnet(train.mat, train.train$SalePrice, alpha = 0,
                     lambda = grid, thresh = 1e-12)
bestlam.ridge = cv.ridge$lambda.min
pred.ridge = predict(fit.ridge, s = bestlam.ridge, 
                     newx = test.mat)
mean((pred.ridge - train.test$SalePrice) ^ 2)
#The test error of ridge increases a lot.

fit.lasso = glmnet(train.mat, train.train$SalePrice, alpha = 1,
                   lambda = grid, thresh = 1e-12)
cv.lasso = cv.glmnet(train.mat, train.train$SalePrice, alpha = 1,
                     lambda = grid, thresh = 1e-12)
bestlam.lasso = cv.lasso$lambda.min
pred.lasso = predict(fit.lasso, s = bestlam.lasso, 
                     newx = test.mat)
mean((pred.lasso - train.test$SalePrice) ^ 2)
#The test error is still high.

test.avg = mean(train.test$SalePrice)
ridge.r2 = 1 - mean((pred.ridge - train.test$SalePrice) ^ 2) / mean((test.avg - train.test$SalePrice) ^ 2)
lasso.r2 = 1 - mean((pred.lasso - train.test$SalePrice) ^ 2) / mean((test.avg - train.test$SalePrice) ^ 2)
rf.r2 = 1 - mean((pred.rf - train.test$SalePrice) ^ 2) / mean((test.avg - train.test$SalePrice) ^ 2)
bag.r2 = 1 - mean((pred.bag - train.test$SalePrice) ^ 2) / mean((test.avg - train.test$SalePrice) ^ 2)
tree.r2 = 1 - mean((pred.tree - train.test$SalePrice) ^ 2) / mean((test.avg - train.test$SalePrice) ^ 2)
#The R square of rf.r2 is the highest.

test$SalePrice = predict(rf.train, newdata = test)
SalePrice = test$SalePrice
sample = data.frame(Id, SalePrice)
write.csv(sample, file = "sample.csv", row.names = FALSE)
