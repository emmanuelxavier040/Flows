
def plot_3d():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import multivariate_normal

    # Define the mean and covariance matrix
    mean = np.array([0, 0, 0])
    covariance = np.eye(3)  # Identity matrix for covariance

    # Create a grid of points
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)

    # Fix the z-coordinate
    Z = np.zeros_like(X)

    # Create the grid of positions for evaluating the PDF
    pos = np.dstack((X, Y, Z)).reshape(-1, 3)

    # Compute the PDF
    rv = multivariate_normal(mean, covariance)
    pdf = rv.pdf(pos)

    # Reshape the PDF to fit the grid shape
    pdf = pdf.reshape(X.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, pdf, cmap='viridis')

    # Add labels
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Density')

    # Show the plot
    plt.show()


# plot_3d()

def plt_3():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import multivariate_normal

    # Parameters for the Gaussian distribution
    mean = np.array([0, 0, 0])  # Mean vector
    covariance = np.eye(3)  # Covariance matrix (identity matrix)

    # Generate grid
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    z = np.linspace(-3, 3, 50)
    X, Y, Z = np.meshgrid(x, y, z)

    # Flatten the grid for evaluation
    positions = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T

    # Define the multivariate normal distribution
    rv = multivariate_normal(mean, covariance)

    # Compute the PDF
    pdf = rv.pdf(positions)

    # Reshape PDF to fit the grid shape
    pdf = pdf.reshape(X.shape)

    # Plotting
    fig = plt.figure(figsize=(12, 8))

    # Plot a surface for a specific slice (e.g., middle slice of z)
    ax = fig.add_subplot(111, projection='3d')
    slice_index = X.shape[2] // 2  # Middle slice

    # Plot surface
    ax.plot_surface(X[:, :, slice_index], Y[:, :, slice_index], pdf[:, :, slice_index], cmap='viridis')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')

    # Show the plot
    plt.show()


def plot():
    import numpy as np

    # Parameters for the Gaussian distribution
    mean = np.array([0, 0, 0])  # Mean vector
    covariance = np.eye(3)  # Covariance matrix (identity matrix)

    # Generate samples
    n_samples = 1000
    samples = np.random.multivariate_normal(mean, covariance, n_samples)
    from scipy.stats import multivariate_normal

    # Define the grid range
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    z = np.linspace(-3, 3, 30)
    X, Y, Z = np.meshgrid(x, y, z)

    # Flatten the grid for evaluation
    positions = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T

    # Define the multivariate normal distribution
    rv = multivariate_normal(mean, covariance)

    # Compute the PDF
    pdf = rv.pdf(positions)

    # Reshape the PDF to fit the grid shape
    pdf = pdf.reshape(X.shape)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot a surface for a specific slice (e.g., middle slice of z)
    slice_index = X.shape[2] // 2  # Middle slice
    ax.plot_surface(X[:, :, slice_index], Y[:, :, slice_index], pdf[:, :, slice_index], cmap='viridis')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')

    plt.show()


plt_3()
