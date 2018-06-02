library(ElemStatLearn)
library(gam)
data(spam)

spam_log = spam
spam_log[,1:57] = log(spam_log[,1:57] + .1)
tr_idx = sample(1:dim(spam)[1], round(2/3 * dim(spam)[1])) # 3067
ts_idx = setdiff(1:dim(spam)[1], tr_idx) # 1534
  
spam_tr = spam_log[tr_idx,]
spam_ts = spam_log[ts_idx,]

# construct formula using cubic smoothing spline with deg = 4 (trace = 5)
df = 3 # 1 linear (without counting intercept) + 3 non-linear
f = 'spam~'
for(i in 1:57){
  f = paste(f, 's(',names(spam)[i],',',as.character(df),')+', sep='')
}
f = substr(f, start=1, stop=nchar(f)-1) 

gam_fit = gam(as.formula(f), family = 'binomial', data = spam_tr)
summary(gam_fit)
gam_pred = predict(gam_fit, newdata = spam_ts, type ='response')
gam_acc = sum((gam_pred > .5) == (spam_ts$spam == 'spam')) / dim(spam_ts)[1]
print(gam_acc)

## Pruning Classification Trees
spam_tr = spam[tr_idx,]
spam_ts = spam[ts_idx,]

tree_fit = tree(spam~., data = spam_tr, split='deviance')
summary(tree_fit) ## Residual mean deviance is minus log likelihood divided by N - size(T)
plot(tree_fit)
text(tree_fit, pretty=0)

cv_tree = cv.tree(tree_fit, FUN = prune.misclass)
best_size = cv_tree$size[which(cv_tree$dev == min(cv_tree$dev))]
par(mfrow=c(1,2))
plot(cv_tree$size ,cv_tree$dev ,type="b") # number of terminal nodes
plot(cv_tree$k ,cv_tree$dev ,type="b") # \alpha in textbook
prune_tree = prune.misclass(tree_fit, best=best_size)
plot(prune_tree)
text(prune_tree, pretty = 0)

tree_pred = predict(prune_tree, newdata = spam_ts, type='class')
tree_acc = sum(tree_pred == spam_ts$spam) / dim(spam_ts)[1]

