import math

import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def plot_loss(loss_values):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(len(loss_values)), y=loss_values, color='blue', marker='o', markersize=5)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_mvn():
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    mean_np = mean.numpy()
    cov_np = cov.numpy()
    rv = multivariate_normal(mean_np, cov_np)

    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)

    positions = np.vstack([X.ravel(), Y.ravel()]).T
    z = rv.pdf(positions)
    z = z.reshape(X.shape)

    plt.contour(X, Y, z)
    plt.title('Multivariate Normal Distribution (Mean: [0, 0], Covariance: [[1, 0.5], [0.5, 1]])')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot():
    mean = [0, 0]  # 2D normal distribution
    cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix

    # Define grid range (adjust as needed)
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3
    grid_size = 100  # Number of points in each dimension

    # Create a mesh grid of points
    X, Y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

    # Evaluate the probability density function (PDF) at each point
    pos = np.dstack((X, Y))  # Combine X and Y into a 3D array for multivariate_normal
    pdf = multivariate_normal(mean, cov).pdf(pos)

    # Marginal distributions along each axis (X and Y)
    marginal_x = np.sum(pdf, axis=0)
    marginal_y = np.sum(pdf, axis=1)

    # Plot the marginal distributions
    plt.figure(figsize=(10, 6))  # Adjust figure size as desired

    # Marginal distribution for X (solid line)
    plt.plot(X[0, :], marginal_x, label='Marginal X', color='blue', linewidth=2)

    # Marginal distribution for Y (dashed line)
    plt.plot(X[0, :], marginal_y, label='Marginal Y', color='orange', linewidth=2, linestyle='--')

    # Set labels and title
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Marginal Distributions of Gaussian Mixture')
    plt.legend()  # Add legend for clarity

    # Show the plot
    plt.show()


def plot2():
    mean = torch.tensor([2.0, 3.0])
    cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])

    # Create a grid of points
    x_min, x_max = mean[0] - 3 * torch.sqrt(cov[0, 0]), mean[0] + 3 * torch.sqrt(cov[0, 0])
    y_min, y_max = mean[1] - 3 * torch.sqrt(cov[1, 1]), mean[1] + 3 * torch.sqrt(cov[1, 1])
    num_points = 200
    x = torch.linspace(x_min, x_max, num_points)
    y = torch.linspace(y_min, y_max, num_points)
    X, Y = torch.meshgrid(x, y)  # Create a meshgrid for x and y

    # Calculate the probability density
    # Create a multivariate normal distribution object
    rv = multivariate_normal(mean, cov)

    # Calculate the probability density at each point
    density = rv.pdf(np.dstack([X, Y]))  # Combine X and Y into a 3D array
    print(density)
    # Marginal distributions
    marginal_x = density.sum(axis=0)  # Sum over y-axis to get marginal for x
    marginal_y = density.sum(axis=1)  # Sum over x-axis to get marginal for y

    # Plot marginals
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(x, marginal_x)
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Marginal Distribution (X)")

    plt.subplot(122)
    plt.plot(y, marginal_y)
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.title("Marginal Distribution (Y)")

    plt.tight_layout()
    plt.show()


# plot2()


# mean = torch.tensor([2.0, 3.0])
# cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
#
# # Create a grid of points
# x_min, x_max = mean[0] - 3 * torch.sqrt(cov[0, 0]), mean[0] + 3 * torch.sqrt(cov[0, 0])
# y_min, y_max = mean[1] - 3 * torch.sqrt(cov[1, 1]), mean[1] + 3 * torch.sqrt(cov[1, 1])
# num_points = 2
# x = torch.linspace(x_min, x_max, num_points)
# y = torch.linspace(y_min, y_max, num_points)
# X, Y = torch.meshgrid(x, y)
# print(X)
# print(Y)
# print(np.dstack([X, Y]))

def plot_analytical_vs_flow_posterior_ridge_regression_fixed_variance_2(mean_np, cov_np,
                                                                      flows):
    x_min, x_max = mean_np[0] - 3 * torch.sqrt(cov_np[0, 0]), mean_np[0] + 3 * torch.sqrt(cov_np[0, 0])
    y_min, y_max = mean_np[1] - 3 * torch.sqrt(cov_np[1, 1]), mean_np[1] + 3 * torch.sqrt(cov_np[1, 1])
    num_points = 5
    x = torch.linspace(x_min, x_max, num_points)
    y = torch.linspace(y_min, y_max, num_points)
    X, Y = torch.meshgrid(x, y)

    rv = multivariate_normal(mean_np, cov_np)
    density = rv.pdf(np.dstack([X, Y]))

    marginal_w_0 = density.sum(axis=0)
    marginal_w_1 = density.sum(axis=1)

    # plt.figure(figsize=(10, 5))
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    plt.subplot(121)
    plt.plot(x, marginal_w_0)  # Use plt.plot for marginal distributions
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Marginal Distribution (X)")

    plt.subplot(122)
    plt.plot(y, marginal_w_1)  # Use plt.plot for marginal distributions
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.title("Marginal Distribution (Y)")

    X = X.unsqueeze(2)
    Y = Y.unsqueeze(2)
    grid_points = torch.cat((X, Y), dim=1)
    grid_points = grid_points.view(-1, 2)

    flow_density = flows.log_prob(grid_points)

    print(flow_density)

    flow_marginal_w_0 = flow_density.sum(axis=0).detach().numpy()

    plt.subplot(121)
    plt.plot(x, flow_marginal_w_0)  # Use plt.plot for marginal distributions
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Marginal Distribution (X)")


    plt.tight_layout()
    plt.show()



def plot_analytical_vs_flow_posterior_ridge_regression_fixed_variance(mean_np, cov_np,
                                                                      flow_samples):
    analytical_samples = multivariate_normal.rvs(mean_np, cov_np, size=50000)
    analytical_samples = pd.DataFrame(analytical_samples)
    analytical_samples["type"] = "analytical"
    flow_samples = pd.DataFrame(flow_samples)
    flow_samples["type"] = "flow"
    data = pd.concat([analytical_samples, flow_samples])
    num_axes = mean_np.size
    fig, axes = plt.subplots(math.ceil(num_axes / 3), 3, figsize=(10, 5))

    for i in range(math.ceil(num_axes / 3) * 3 - num_axes):
        axes[math.ceil(num_axes / 3) - 1, 3 - i - 1].axis('off')

    for i, ax in enumerate(axes.flat):
        if ax.axison:
            sns.histplot(data=data, x=i, ax=ax, hue="type", kde=True, bins=100)
            ax.legend(['analytical', 'flow'])
            ax.set_xlabel("W_" + str(i + 1))
            ax.set_ylabel("Value")

    plt.suptitle('Analytical Vs Flow Posterior - Ridge Regression')
    plt.tight_layout()
    plt.show()


def plot_analytical_vs_learnt_posterior_Ridge_Regression_student_t(X, Y, prior_mean, prior_covariance, a_0, b_0):
    return 0


def plot_mvn_2(data):
    # mean = torch.tensor([0.0, 0.0])
    # cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    # mean_np = mean.numpy()
    # cov_np = cov.numpy()
    # data = multivariate_normal.rvs(mean_np, cov_np, size=1000)
    # sns.kdeplot(data=data)
    data = pd.DataFrame(data)
    sns.jointplot(
        data=data,
        kind="kde"
    )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Multivariate Normal Distribution (Seaborn KDE)')
    plt.show()
