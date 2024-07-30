import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generate sample data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a range of lambda values
lambdas = np.logspace(-4, 1, 100)

# Store coefficients and L1 norms for each lambda
coefs = []
l1_norms = []

for alpha in lambdas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_scaled, y)
    coefs.append(lasso.coef_)
    l1_norms.append(np.sum(np.abs(lasso.coef_)))

coefs = np.array(coefs)
l1_norms = np.array(l1_norms)

# Normalize the L1 norms
relative_l1_norms = l1_norms / l1_norms.max()
# Plotting the coefficients against the relative L1 norm
plt.figure(figsize=(10, 8))

# Iterate over the coefficients for the first few features for visibility
for i in range(coefs.shape[1]):
    plt.plot(relative_l1_norms, coefs[:, i], label=f'Beta {i + 1}')

# Labeling the plot
plt.xlabel('Relative L1 Norm of Betas')
plt.ylabel('Beta Coefficients')
plt.title('Lasso Paths: Betas vs. Relative L1 Norm of Betas')
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(True)
plt.show()
