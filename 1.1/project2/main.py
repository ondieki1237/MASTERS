import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def pca_with_svd(data, k=2):
    """
    Perform PCA using SVD for dimensionality reduction.
    
    Args:
        data: Centered data matrix (n_samples x n_features)
        k: Number of principal components to keep
    
    Returns:
        projected_data: Data projected onto k principal components
    """
    # Perform Singular Value Decomposition (SVD)
    U, sigma, VT = np.linalg.svd(data, full_matrices=False)

    # Select the top k singular values and corresponding vectors
    # VT contains the principal components (eigenvectors) as rows
    # We take the first k rows (top k principal components)
    top_k_components = VT[:k, :]

    # Project data into k-dimensional space
    # projected_data = data @ top_k_components.T
    projected_data = data @ top_k_components.T

    return projected_data

def plot_2d_projection(projected_data, labels):
    """
    Visualize the 2D projection of the data.
    
    Args:
        projected_data: Data projected onto 2 principal components
        labels: Class labels for coloring the points
    """
    plt.figure(figsize=(8, 6))
    
    # Create a scatter plot
    plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=100)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D Projection of Iris Dataset (PCA with SVD)")
    plt.colorbar(label='Species')
    
    # Save the plot to a file
    plt.savefig('iris_pca_2d_projection.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'iris_pca_2d_projection.png'")
    
    # Optionally show the plot (comment out if running in headless environment)
    # plt.show()

# Load the Iris dataset
iris = load_iris()
data = iris.data
labels = iris.target

# Step 1: Center the data (subtract the mean of each feature)
data_centered = data - np.mean(data, axis=0)

print("=" * 70)
print("PCA using Singular Value Decomposition (SVD)")
print("=" * 70)
print(f"\nOriginal data shape: {data.shape}")
print(f"Centered data shape: {data_centered.shape}")

# Step 2: Perform PCA using SVD
projected_data = pca_with_svd(data_centered, k=2)

print(f"\nProjected data shape: {projected_data.shape}")
print("\nExplanation:")
print("-" * 70)
print("How SVD is applied in PCA:")
print("1. Center the data by subtracting the mean from each feature")
print("2. Perform SVD: X = U * Σ * V^T")
print("   - U: Left singular vectors (n_samples x n_samples)")
print("   - Σ: Singular values (diagonal matrix)")
print("   - V^T: Right singular vectors (n_features x n_features)")
print("3. The rows of V^T are the principal components (eigenvectors)")
print("4. The singular values in Σ are related to eigenvalues")
print("5. Project data onto top k components: X_reduced = X * V[:k]^T")
print("-" * 70)

# Step 3: Visualize the 2D projection
plot_2d_projection(projected_data, labels)
