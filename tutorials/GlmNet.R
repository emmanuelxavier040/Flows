library(glmnet)

x = matrix(rnorm(100 * 20), 100, 20)
y = rnorm(100)
fit1 = glmnet(x, y, lambda=c(2, 3, 1, 5, 7, 3, 8, 6, 4, 9))
# print(fit1)
cv.glmnet( x, y, type.measure = "mse", alpha=1)
# coef(fit1, s = 0.01) # extract coefficients at a single value of lambda
# predict(fit1, newx = x[1:10, ], s = c(0.01, 0.005)) # make predictions
#
# # Relaxed
# fit1r = glmnet(x, y, relax = TRUE) # can be used with any model
#



data(PoissonExample)
x <- PoissonExample$x
y <- PoissonExample$y
fit <- glmnet(x, y, family = "poisson", lambda=c(2, 3, 1, 5, 7, 3, 8, 6, 4, 9))
plot(fit)