library(grpreg)

# graphics.off()  # clear all graphs
# rm(list = ls()) # remove all files from your workspace
#
# set.seed(1234)
#
# # Birthweight data
# data(Birthwt)
# X <- Birthwt$X
# group <- Birthwt$group
#
# print("Groups")
# print(group)
#
#
# # Linear regression
# y <- Birthwt$bwt
# fit <- grpreg(X, y, group, family="poisson", penalty="grLasso")
# plot(fit)
#
#
# select(fit, "AIC")

# Simulate some example data
set.seed(123)
n <- 100  # Number of observations
p <- 10   # Number of predictors
X <- matrix(rnorm(n * p), n, p)  # Predictor matrix
y <- rpois(n, lambda = exp(X[, 1] * 0.5 + X[, 2] * 0.3))  # Poisson response

# Define the group structure for the variables
group <- rep(1:5, each = 2)  # Grouping two variables into each of 5 groups

# Fit the group lasso Poisson regression model
fit <- grpreg(X, y, group = group, family = "poisson", penalty="grLasso")

# Perform cross-validation to find the optimal lambda
cv_fit <- cv.grpreg(X, y, group = group, family = "poisson", penalty="grLasso")

# Print the optimal lambda value
cat("Optimal lambda:", cv_fit$lambda.min, "\n")

# Get the coefficients for the model with the best lambda
best_coefs <- coef(cv_fit, lambda = cv_fit$lambda.min)

# Print the coefficients
print(best_coefs)

# Plot the cross-validation curve
plot(cv_fit)

# Plot the solution path for different lambda values
plot(fit)