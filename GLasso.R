library(gglasso)
library(stats)
library(glmnet)

GLassoPath <- function(X, y, group, lambdas, loss="ls") {
    fit <- gglasso(x = X, y = y, group = group, loss = loss, lambda=lambdas, maxit=3e8)
    return (fit)
}

CVGLasso <- function(X, y, group, lambdas) {
    cv <- cv.gglasso(x=X, y=y, group=group, lambda=lambdas,loss="ls", pred.loss="L2",
            intercept = F, nfolds=9)
    return (cv)
}

Glm <- function(X, Y, data) {
    cv <- glm(Y ~ X, family = poisson(link = "log"), data = data)
    return (cv)
}

Glmnet <- function(X, Y, lambdas, alpha, family="gaussian") {
    fit <- glmnet(X, Y, lambda=lambdas, alpha=alpha, family=family, intercept=FALSE)
    return (fit)
}

CVGlmnet <- function(X, Y, lambdas, alpha) {
    fit <- cv.glmnet(X, Y, lambda=lambdas, type.measure = "mse", alpha=alpha)
    return (fit)
}

