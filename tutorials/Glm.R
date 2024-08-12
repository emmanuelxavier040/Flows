library(stats)

# Example predictor matrix (X) with 3 predictors and 10 observations
X <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
            nrow = 10, ncol = 3)

# Example response vector (Y) with 10 observations
Y <- c(2, 3, 1, 5, 7, 3, 8, 6, 4, 9)

# Convert the matrices to a data frame
data <- data.frame(Y = Y, X)

# Fit a Poisson regression model
model <- glm(Y ~ ., family = poisson(link = "log"), data = data)

# Summary of the model
summary(model)

