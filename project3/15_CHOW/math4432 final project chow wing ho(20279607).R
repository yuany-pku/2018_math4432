rm(list=ls())
library('ggplot2') 
library('ggthemes') 
library('scales') 
library('dplyr') 
library('mice') 
library('randomForest') 
library("party")
train <- read.csv("tittrain.csv", stringsAsFactors = F)
test  <- read.csv("tittest.csv", stringsAsFactors = F)
full  <- bind_rows(train, test) 
str(full)

full$Survived <- factor(full$Survived)
ggplot(data = full[1:nrow(train),],aes(x = Pclass, y = ..count.., fill=Survived)) + 
  geom_bar(stat = "count", position='dodge') + 
  xlab('pclass') + 
  ylab('passagers') + 
  scale_fill_manual(values = c("red","green")) + 
  theme_economist(base_size=16)+
  geom_text(stat = "count", aes(label = ..count..), position=position_dodge(width=1), vjust=-0.5)

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title)
special_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% special_title]  <- 'special'

table(full$Sex, full$Title)
ggplot(data = full[1:891,], mapping = aes(x = Title, y = ..count.., fill=Survived)) + 
  geom_bar(stat = "count", position='stack') + 
  xlab('title') + 
  ylab('passagers') + 
  geom_text(stat = "count", aes(label = ..count..), position=position_stack(vjust = 0.5)) +
  scale_fill_wsj() + theme_economist(base_size=16)


full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])


full$Fsize <- full$SibSp + full$Parch + 1


full$Family <- paste(full$Surname, full$Fsize, sep='_')

full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

ggplot(data = full[1:nrow(train),], mapping = aes(x = Fsize, y = ..count.., fill=Survived)) + 
  geom_bar(stat = 'count', position='dodge') + 
  xlab('family member') + 
  ylab('passagers') + 
  geom_text(stat = "count", aes(label = ..count..), position=position_dodge(width=1), vjust=-0.5) + 
  scale_fill_wsj() + theme_economist(base_size=12)

ggplot(full[1:nrow(train), ], mapping = aes(x = Embarked, y = ..count.., fill = Survived)) +
  geom_bar(stat = 'count', position='dodge') + 
  xlab('embarked') +
  ylab('passagers') +
  geom_text(stat = "count", aes(label = ..count..), position=position_dodge(width=1), vjust=-0.5) + 
  scale_fill_wsj() + theme_economist(base_size=12)

full$Cabin[1:28]
strsplit(full$Cabin[2], NULL)[[1]]
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

full[c(62, 830), 'Embarked']
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)


full$Embarked[c(62, 830)] <- 'C'
full[1044, ]
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)
sum(is.na(full$Age))

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
set.seed(129)
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original ', 
     col='red', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='green', ylim=c(0,0.04))


full$Age <- mice_output$Age
sum(is.na(full$Age))
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'
table(full$Child, full$Survived)
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'
table(full$Mother, full$Survived)
full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)
md.pattern(full)

train <- full[1:891,]
test <- full[892:1309,]
set.seed(754)
rfmodel <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)
importance    <- importance(rfmodel)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))


rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
prediction <- predict(rfmodel, test)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(solution, file = '155.csv', row.names = F)
ticket.count <- aggregate(full$Ticket, by = list(full$Ticket), function(x) sum(!is.na(x)))
full$TicketCount <- apply(full, 1, function(x) ticket.count[which(ticket.count[, 1] == x['Ticket']), 2])
full$TicketCount <- factor(sapply(full$TicketCount, function(x) ifelse(x > 1, 'share', 'unique')))
full$Fsize<-as.numeric(full$Fsize)
full$Surname <- sapply(full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
full$FamilyID <- paste(as.character(full$Fsize), full$Surname, sep="")
full$FamilyID[full$Fsize <= 2] <- 'Small'
famIDs <- data.frame(table(full$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
full$FamilyID[full$FamilyID %in% famIDs$Var1] <- 'Small'
full$FamilyID <- factor(full$FamilyID)
train <- full[1:891,]
test <- full[892:1309,]
set.seed(415)
model <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + Title + Fsize + FamilyID +
                 TicketCount, data = full[1:nrow(train), ], controls=cforest_unbiased(ntree=2000, mtry=3))
predict.result <- predict(model, test, OOB=TRUE, type = "response")
solution <- data.frame(PassengerID = test$PassengerId, Survived = predict.result)
write.csv(solution, file = 'fdg.csv', row.names = F)

