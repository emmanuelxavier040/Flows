library(grplasso)

data(splice)

contr <- rep(list("contr.sum"), ncol(splice) - 1)
names(contr) <- names(splice)[-1]

fit <- grplasso(y ~ ., data = splice, model = PoissReg(), lambda = 10,
                contrasts = contr, standardize = TRUE, center = TRUE)
print(fit)
pred <- predict(fit)
pred.resp <- predict(fit, type = "response")

## The following points should lie on the sigmoid curve
plot(pred, pred.resp)