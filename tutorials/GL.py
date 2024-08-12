import numpy as np
from group_lasso import GroupLasso

# Step 1: Define the Group Structure
# Let's assume we have 10 features divided into 3 groups
groups = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]

# Step 2: Generate Features
np.random.seed(0)  # For reproducibility
n_samples = 100
n_features = 10
X = np.random.randn(n_samples, n_features)

# Step 3: Generate Coefficients
# Let's create true coefficients such that within-group coefficients are related
true_coefs = np.zeros(n_features)
true_coefs[:2] = [1, -1]  # Group 0
true_coefs[2:5] = [0.5, -0.5, 0.5]  # Group 1
true_coefs[5:] = [1, 1, -1, -1, 0.5]  # Group 2

# Step 4: Generate Response Variable
# Compute y = X @ true_coefs + noise
noise = np.random.randn(n_samples) * 0.1
y = X @ true_coefs + noise

print("Generated features (X):\n", X[:5])
print("True coefficients (true_coefs):\n", true_coefs)
print("Generated response variable (y):\n", y[:5])



# Define group sizes
group_lasso = GroupLasso(groups=groups, group_reg=0.1, l1_reg=0.1, n_iter=1000, old_regularisation=True)

# Fit the model
group_lasso.fit(X, y)

# Print the learned coefficients
print("Learned coefficients:\n", group_lasso.coef_)
