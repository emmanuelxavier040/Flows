library(grpreg)

graphics.off()  # clear all graphs
rm(list = ls()) # remove all files from your workspace

set.seed(1234)

# Birthweight data
data(Birthwt)
X <- Birthwt$X
group <- Birthwt$group

print("Groups")
print(group)


# Linear regression
y <- Birthwt$bwt
fit <- grpreg(X, y, group, penalty="grLasso")
plot(fit)


select(fit, "AIC")

