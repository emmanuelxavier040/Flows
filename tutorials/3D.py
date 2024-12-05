import matplotlib.pyplot as plt
import torch
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
#
# Define the grid and complex distribution
sample_size = 50
X = np.linspace(-5, 5, sample_size)
Y = np.linspace(-5, 5, sample_size)
X, Y = np.meshgrid(X, Y)

# Define the complex distribution
Z = np.sin(np.sqrt(X ** 2 + Y ** 2)) + np.cos(X) * np.sin(Y)

# Flatten the grid arrays for easier handling
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Set up the figure and axis for the 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Complex 3D Distribution Surface', fontsize=18)
ax.set_xlabel(r'$\beta_1$', fontsize=18)
ax.set_ylabel(r'$\beta_2$', fontsize=18)
ax.set_zticks([])
ax.set_zlabel("Probability", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# #
# # # Set up the 2D scatter plot
# # fig, ax = plt.subplots(figsize=(8, 6))
# # scatter = ax.scatter([], [], c=[], s=100, alpha=0.6, cmap='viridis')
# # ax.set_title('Simulated Gibbs Sampling Scatter Plot', fontsize=14)
# # ax.set_xlabel(r'$\beta_1$', fontsize=18)
# # ax.set_ylabel(r'$\beta_2$', fontsize=18)
# # plt.xticks(fontsize=15)
# # plt.yticks(fontsize=15)
# # ax.set_xlim(-5, 5)
# # ax.set_ylim(-5, 5)
# # plt.grid(True)
# #
# #
# # # Function to simulate multiple point selections
# # def select_points(num_points=5):
# #     probabilities = np.exp(Z_flat - np.max(Z_flat))  # Exponential for probability weighting
# #     probabilities /= probabilities.sum()  # Normalize to get a valid probability distribution
# #     chosen_indices = np.random.choice(len(Z_flat), size=num_points, p=probabilities)  # Sample multiple indices
# #     return X_flat[chosen_indices], Y_flat[chosen_indices], Z_flat[chosen_indices]  # Return chosen x, y, z
# #
# #
# # # Prepare to save frames for the GIF
# # frames = []
# #
# #
# # # Animation update function
# # def update(frame):
# #     x, y, z = select_points(num_points=10)  # Select multiple points
# #     scatter.set_offsets(
# #         np.append(scatter.get_offsets(), [[x[i], y[i]] for i in range(len(x))], axis=0))  # Update scatter offsets
# #     scatter.set_array(np.append(scatter.get_array(), z))  # Update the colors based on Z value
# #
# #     # Save the current frame as an image
# #     plt.draw()  # Ensure the plot is updated
# #     frame_file = f"frame_{frame}.png"
# #     plt.savefig(frame_file)  # Save the current frame
# #     frames.append(frame_file)  # Append the filename to the frames list
# #     return scatter,
# #
# #
# # # Create the animation
# # ani = FuncAnimation(fig, update, frames=1000, interval=100, blit=False)  # Reduced interval for faster updates
# #
# # # Display the plot
# # plt.show()
# #
# # # Create the GIF from the saved frames
# # with imageio.get_writer('gibbs_sampling_simulation.gif', mode='I', duration=0.1) as writer:
# #     for frame in frames:
# #         image = imageio.imread(frame)
# #         writer.append_data(image)
# #
# # # Optionally, clean up by removing the saved frame images
# # import os
# #
# # for frame in frames:
# #     os.remove(frame)
# #
# # print("GIF created successfully!")
# #==================================================
#
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
#
#
# mu1_x = -1
# mu1_y = 1
# sigma1 = 0.5
#
# mu2 = np.array([2, -1])
# cov2 = np.array([[0.8, 0.2], [0.2, 0.5]])
#
# dimensions = 2
# sample_size = 50
#
# x = torch.linspace(-3, 5, sample_size)
# y = torch.linspace(-3, 3, sample_size)
# X, Y = torch.meshgrid(x, y, indexing='ij')
#
# X = X.float()
# Y = Y.float()
#
# Z1 = (1 / (2 * np.pi * sigma1**2)) * torch.exp(-0.5 * (((X - mu1_x) ** 2 + (Y - mu1_y) ** 2) / sigma1**2))
#
# pos = torch.dstack((X, Y))
# cov2_inv = torch.tensor(np.linalg.inv(cov2), dtype=torch.float32)
# diff = pos - torch.tensor(mu2, dtype=torch.float32)
#
# Z2 = (1 / (2 * np.pi * np.sqrt(np.linalg.det(cov2)))) * torch.exp(
#     -0.5 * torch.einsum('...i,ij,...j', diff, cov2_inv, diff)
# )
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# ax.plot_surface(X.numpy(), Y.numpy(), Z1.numpy(), cmap='viridis', alpha=0.6, label='Distribution 1')
# ax.plot_surface(X.numpy(), Y.numpy(), Z2.numpy(), cmap='plasma', alpha=0.6, label='Distribution 2')
#
# ax.set_xlabel(r'$\beta_1$', fontsize=18)
# ax.set_ylabel(r'$\beta_2$', fontsize=18)
# ax.set_zlabel("Probability Density", fontsize=18)
#
# ax.set_title('2D Gaussian Distributions (Isotropic and Multivariate)', fontsize=20)
#
# ax.view_init(elev=30, azim=210)
# plt.show()
#

#=============================================
import seaborn as sns
sns.set_theme(style="darkgrid")
dimensions = 2
sample_size = 50

# mu_x = 2
# mu_y = 1
# x = torch.linspace(-1.5, 5.5, sample_size)
# y = torch.linspace(-2.5, 4.5, sample_size)

mu_x = 0
mu_y = 0
x = torch.linspace(-3, 3, sample_size)
y = torch.linspace(-3, 3, sample_size)

X, Y = torch.meshgrid(x, y, indexing='ij')
flow_density_reshaped = (1 / (2 * torch.pi)) * torch.exp(-0.5 * ((Y - mu_y)**2 + (X - mu_x)**2))
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

X_np = X.detach().cpu().numpy()
Y_np = Y.detach().cpu().numpy()
flow_density_reshaped_np = flow_density_reshaped.detach().cpu().numpy()

# ax.plot_surface(X_np, Y_np, flow_density_reshaped_np, cmap='viridis', linewidth=0,
#                 alpha=1.0,
#                 rcount=sample_size,
#                 ccount=sample_size)
ax.plot_surface(X_np, Y_np, flow_density_reshaped_np, cmap='Blues', alpha=0.4, rstride=1, cstride=1, edgecolor='black')

ax.set_xlabel(r'$\beta_1$', fontsize=18)
ax.set_ylabel(r'$\beta_2$', fontsize=18)
ax.set_zticks([])
ax.set_zlabel("Probability", fontsize=18)

# ax.set_zticklabels([])
# ax.zaxis.label.set_visible(False)
plt.tight_layout()
plt.margins(x=0, y=0)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.close()


sample_size = 50

X = np.linspace(-5, 5, sample_size)
Y = np.linspace(-5, 5, sample_size)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2)) + np.cos(X) * np.sin(Y)
# Z = ((1 / (2 * np.pi)) * np.exp(-0.5 * ((Y - mu_y)**2 + (X - mu_x)**2))
#      + 10*(1 / (2 * np.pi)) * np.exp(-0.5 * ((Y - 1)**2 + (X - 1)**2))
#      - 1.5 *(1 / (2 * np.pi)) * np.exp(-0.5 * ((Y - 10)**2 + (X - 5)**2)))


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Complex 3D Distribution Surface', fontsize=18)
ax.set_xlabel(r'$\beta_1$', fontsize=18)
ax.set_ylabel(r'$\beta_2$', fontsize=18)
ax.set_zticks([])
ax.set_zlabel("Probability", fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))

X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
scatter = ax.scatter(X_flat, Y_flat, c=Z_flat,   s=100)
ax.set_title('Scatter Plot on X, Y Plane with Z as Color Intensity', fontsize=14)
ax.set_xlabel(r'$\beta_1$', fontsize=18)
ax.set_ylabel(r'$\beta_2$', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()




import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Beta mixture model parameters for 4 distinct peaks
weights = [0.25, 0.25, 0.25, 0.25]  # Mixing coefficients, must sum to 1
alpha_params = [2, 5, 18, 15]  # Alpha parameters for the 4 Beta distributions
beta_params = [8, 3, 12, 5]    # Beta parameters for the 4 Beta distributions

# Function to calculate the Beta PDF manually
def beta_pdf_manual(x, alpha, beta):
    # Avoid division by zero by ensuring x is within (0,1)
    if x <= 0 or x >= 1:
        return 0
    # Compute the Beta function B(alpha, beta)
    B_ab = (gamma(alpha) * gamma(beta)) / gamma(alpha + beta)
    # Return the Beta PDF
    return (x**(alpha - 1) * (1 - x)**(beta - 1)) / B_ab

# Function to calculate the PDF of the Beta Mixture Model at a given point (x, y)
def beta_mixture_pdf_manual(x, y, weights, alpha_params, beta_params):
    pdf = 0
    for weight, alpha, beta_param in zip(weights, alpha_params, beta_params):
        # Calculate the joint PDF as the product of two Beta PDFs (x and y), assuming independence
        pdf += weight * beta_pdf_manual(x, alpha, beta_param) * beta_pdf_manual(y, alpha, beta_param)
    return pdf

# Define the range of x and y values (since Beta is typically defined between 0 and 1)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Calculate the PDF for each (x, y) pair
Z = np.array([beta_mixture_pdf_manual(x_val, y_val, weights, alpha_params, beta_params) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
Z = Z.reshape(X.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis')

# Labels and Title
ax.set_title('3D Beta Mixture Model (4 Distinct Peaks, Manual PDF)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability Density')

plt.show()
