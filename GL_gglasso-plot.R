library(gglasso)

# load bardet data set
# data(bardet)
#
# group1 <- rep(1:20, each = 5)
#
# m1 <- gglasso(x = bardet$x, y = bardet$y, group = group1, loss = "ls")
#
# plot(m1)
# plot(m1,group=TRUE) # plots group norm against the log-lambda sequence
# plot(m1,log.l=FALSE)

library(MASS)  # For mvrnorm to generate multivariate normal data


n <- 1000              # Number of observations
p <- 20                 # Number of predictors (can be changed dynamically)
beta <- rnorm(p)       # Coefficients vector of length p, randomly generated
sigma <- 1             # Standard deviation of the noise

# Generate X matrix with random normal values
set.seed(42)  # For reproducibility
X <- matrix(rnorm(n * p), nrow = n, ncol = p)

# Generate noise
noise <- rnorm(n, mean = 0, sd = sigma)

# Calculate Y = X * beta + noise
Y <- X %*% beta + noise

group1 <- rep(1:5, each = 4)


m1 <- gglasso(x = X, y = Y, group = group1, loss = "ls")
plot(m1)
plot(m1,group=TRUE) # plots group norm against the log-lambda sequence
plot(m1,group=TRUE, log.l=FALSE)
plot(m1,log.l=FALSE)
