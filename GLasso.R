library(gglasso)


GLassoPath <- function(X, y, group, lambdas, loss="ls") {
    fit <- gglasso(x = X, y = y, group = group, loss = loss, lambda=lambdas, maxit=3e8)
    return (fit)
}

CVLasso <- function(X, y, group, lambdas) {
    cv <- cv.gglasso(x=X, y=y, group=group, lambda=lambdas,loss="ls", pred.loss="L2",
            intercept = F, nfolds=5)
    return (cv)
}