library(glmnet)
library(gglasso)

graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace
 
set.seed(1234)
 
#——————————————–
# X and y variable
#——————————————–
 
N = 500 # number of observations
p = 20  # number of variables
 
# random generated X
X = matrix(rnorm(N*p), ncol=p)
 
# standardization : mean = 0, std=1
X = scale(X)
 
# artificial coefficients
beta = c(0.15, -0.33,0.25, -0.25,0.05,0,0,0,0.5,0.2,
        -0.25, 0.12, -0.125,0,0,0,0,0,0,0)
 
# Y variable, standardized Y
y = X%*%beta + rnorm(N, sd=0.5)
#y = scale(y)

# group index for X variables
v.group <- c(1,1,1,1,1,2,2,2,2,2,
             3,3,3,3,3,4,4,4,4,4)
 
#——————————————–
# Model with a given lambda
#——————————————–
# group lasso
gr <- gglasso(X, y, lambda = 0.2,
             group = v.group, loss="ls",
             intercept = F)

# Results
df.comp <- data.frame(
    group = v.group, beta = beta,
    Group     = gr$beta[,1]
)
# df.comp

#————————————————
# Run cross-validation & select lambda
#————————————————
# lambda.min : minimal MSE
# lambda.1se : the largest λ at which the MSE is
#   within one standard error of the minimal MSE.


# group lasso
print("Hi ====================")
print(X)
paste(y)
gr_cv <- cv.gglasso(x=X, y=y, group=v.group,
            loss="ls", pred.loss="L2",
            intercept = F, nfolds=5)
# x11(); plot(gr_cv)
# dev.new();plot(gr_cv)
paste(gr_cv$lambda.min, gr_cv$lambda.1se)
#
#
#
# #——————————————–
# # Model with selected lambda
# #——————————————–
# # group lasso
# gr <- gglasso(X, y, lambda = gr_cv$lambda.1se+0.1,
#              group = v.group, loss="ls",
#              intercept = F)
# # Results
# df.comp.lambda.1se <- data.frame(
#     group = v.group, beta = beta,
#     Group     = gr$beta[,1]
# )
# df.comp.lambda.1se