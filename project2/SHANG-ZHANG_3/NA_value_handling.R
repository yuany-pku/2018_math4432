#This is a function for handling NA values in house sale price dataset
NA_value_handling = function(modified_data)
{
  library(randomForest)
  #Alley
  modified_data$Alley = as.character(modified_data$Alley)
  modified_data$Alley[is.na(modified_data$Alley)]= 'No Alley'
  modified_data$Alley = as.factor(modified_data$Alley)
  #BsmtQual
  modified_data$BsmtQual = as.character(modified_data$BsmtQual)
  modified_data$BsmtQual[is.na(modified_data$BsmtQual)] = 'No Basement'
  modified_data$BsmtQual = as.factor(modified_data$BsmtQual)
  #BsmtCond
  modified_data$BsmtCond = as.character(modified_data$BsmtCond)
  modified_data$BsmtCond[is.na(modified_data$BsmtCond)] = 'No Basement'
  modified_data$BsmtCond = as.factor(modified_data$BsmtCond)
  #BsmtExposure
  modified_data$BsmtExposure = as.character(modified_data$BsmtExposure)
  modified_data$BsmtExposure[is.na(modified_data$BsmtExposure)] = 'No Basement'
  modified_data$BsmtExposure = as.factor(modified_data$BsmtExposure)
  #BsmtFinType1
  modified_data$BsmtFinType1 = as.character(modified_data$BsmtFinType1)
  modified_data$BsmtFinType1[is.na(modified_data$BsmtFinType1)] = 'No Basement'
  modified_data$BsmtFinType1 = as.factor(modified_data$BsmtFinType1)
  #BsmtFinType2
  modified_data$BsmtFinType2 = as.character(modified_data$BsmtFinType2)
  modified_data$BsmtFinType2[is.na(modified_data$BsmtFinType2)] = 'No Basement'
  modified_data$BsmtFinType2 = as.factor(modified_data$BsmtFinType2)
  #FireplaceQU
  modified_data$FireplaceQu = as.character(modified_data$FireplaceQu)
  modified_data$FireplaceQu[is.na(modified_data$FireplaceQu)] = 'No Fireplace'
  modified_data$FireplaceQu = as.factor(modified_data$FireplaceQu)
  #GarageType
  modified_data$GarageType = as.character(modified_data$GarageType)
  modified_data$GarageType[is.na(modified_data$GarageType)] = 'No Garage'
  modified_data$GarageType = as.factor(modified_data$GarageType)
  #GarageFinish
  modified_data$GarageFinish = as.character(modified_data$GarageFinish)
  modified_data$GarageFinish[is.na(modified_data$GarageFinish)] = 'No Garage'
  modified_data$GarageFinish = as.factor(modified_data$GarageFinish)
  #GarageQual
  modified_data$GarageQual = as.character(modified_data$GarageQual)
  modified_data$GarageQual[is.na(modified_data$GarageQual)] = 'No Garage'
  modified_data$GarageQual = as.factor(modified_data$GarageQual)
  #GarageCond
  modified_data$GarageCond = as.character(modified_data$GarageCond)
  modified_data$GarageCond[is.na(modified_data$GarageCond)] = 'No Garage'
  modified_data$GarageCond = as.factor(modified_data$GarageCond)
  #GarageYrBlt
  for (n in 1:nrow(modified_data))
  {
    if(is.na(modified_data$GarageYrBlt[n])){modified_data$GarageYrBlt[n] = 'No Garage'}
    else if(modified_data$GarageYrBlt[n]>=2002){modified_data$GarageYrBlt[n] = 'New'}
    else if(modified_data$GarageYrBlt[n]>=1980){modified_data$GarageYrBlt[n] = 'Relatively New'}
    else if(modified_data$GarageYrBlt[n]>=1961){modified_data$GarageYrBlt[n] = 'Relatively Old'}
    else {modified_data$GarageYrBlt[n] = 'Old'}
  }
  modified_data$GarageYrBlt = as.factor(modified_data$GarageYrBlt)
  #PoolQC
  modified_data$PoolQC = as.character(modified_data$PoolQC)
  modified_data$PoolQC[is.na(modified_data$PoolQC)] = 'No Pool'
  modified_data$PoolQC = as.factor(modified_data$PoolQC)
  #Fence
  modified_data$Fence = as.character(modified_data$Fence)
  modified_data$Fence[is.na(modified_data$Fence)] = 'No Fence'
  modified_data$Fence = as.factor(modified_data$Fence)
  #MiscFeature
  modified_data$MiscFeature = as.character(modified_data$MiscFeature)
  modified_data$MiscFeature[is.na(modified_data$MiscFeature)] = 'None'
  modified_data$MiscFeature = as.factor(modified_data$MiscFeature)
  #LotFrontage
  LotFrontageAva = rep('Yes',nrow(modified_data))
  LotFrontageAva[is.na(modified_data$LotFrontage)] = 'No'
  modified_data = data.frame(LotFrontageAva,modified_data)
  #Use na.roughfix to fix all other predictors with NA values except LotFrontage
  modified_data = na.roughfix(modified_data)
  modified_data$LotFrontage[modified_data$LotFrontageAva == 'No'] = NA
  
  return(modified_data)
}  