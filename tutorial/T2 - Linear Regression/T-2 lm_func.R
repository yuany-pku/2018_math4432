library(MASS)
library(ISLR)
data(Boston)
?Boston
lm.fit=lm(medv ~ lstat, data=Boston)
names(lm.fit)
summary(lm.fit)

## concrete code
Xtr = cbind(rep(1,506), Boston$lstat)
inv_mat = solve(t(Xtr) %*% Xtr)
beta = inv_mat %*% t(Xtr) %*% Boston$medv
y_pred = Xtr %*% beta
res = Boston$medv - y_pred
quantile(res, c(0,.25,.5,.75,1)) # Res Quantiles

sig_eps = sqrt( sum(res^2) / (dim(Xtr)[1] - 2) ) # Std.Error of Residual
sig_beta0 = sqrt(inv_mat[1,1]) * sig_eps # Std.Error of Intercept(beta0)
sig_beta1 = sqrt(inv_mat[2,2]) * sig_eps # Std.Error of lstat(beta1)

tval_beta0 = beta[1]/sig_beta0 # t-value(Z-score) of intercept
tval_beta1 = beta[2]/sig_beta1 # t-value(Z-score) of lstat
pval_beta0 = 2 * (1 - pt(q = tval_beta0, df = 504)) # p-value of intercept = 0
pval_beta1 = 2 * (pt(q = tval_beta1, df = 504)) # p-value of lstat = 0

# R squared stat, explained var
RSS = sum((y_pred - Boston$medv)**2)
TSS = sum((Boston$medv - mean(Boston$medv))**2 )
R2 = 1 - RSS / TSS
# adjuested R squared, penalizing the number of variables
R2_adjusted = 1 - (1 - R2) * (506-1)/(506-3)

Fstat = (TSS - RSS) / (RSS/(506-2)) # F-statistics
pval_F = 1 - pf(q = Fstat, df1=1, df2=(506-2))# p-value of lstat = 0
