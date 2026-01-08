import numpy as np
import matplotlib.pyplot as plt

# Generate correlated 2D data similar to the image
np.random.seed(42)
n = 5000

# Create elongated diagonal spread similar to the image
theta = np.pi / 6  # angle of main spread
stretch = 3
data_raw = np.random.randn(n, 2)
data_raw[:, 0] *= stretch  # stretch along x
# Rotate
rotation = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
data = data_raw @ rotation.T
# Shift center up
data[:, 1] += 3

# Compute PCA (eigenvectors of covariance)
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Center of data
center = np.mean(data, axis=0)

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot - use blue dots with edge to make them clearly visible as data points
ax.scatter(data[:, 0], data[:, 1], c='steelblue', alpha=0.4, s=15, 
           edgecolors='darkblue', linewidths=0.3, label='Data points')

# Draw eigenvector arrows (scaled by sqrt of eigenvalue for visualization)
scale = 2.0
for i in range(2):
    vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * scale
    label = f'PC{i+1}' if i == 0 else f'PC{i+1}'
    ax.annotate('', xy=(center[0] + vec[0], center[1] + vec[1]), 
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))

# Set axis properties
ax.set_xlim(-8, 10)
ax.set_ylim(-6, 12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)

# Labels
ax.set_xlabel('$x_1$', fontsize=14)
ax.set_ylabel('$x_2$', fontsize=14)
ax.set_title('PCA Demonstration: Principal Component Directions', fontsize=13, fontweight='bold')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', 
                          markersize=8, label='Data points ($n=5000$)'),
                   Line2D([0], [0], color='red', lw=2.5, label='Principal components')]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('pca_demonstration.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pca_demonstration.png")

# Print info
print(f"\nData center: ({center[0]:.2f}, {center[1]:.2f})")
print(f"Eigenvalue 1: {eigenvalues[0]:.3f}")
print(f"Eigenvalue 2: {eigenvalues[1]:.3f}")
print(f"PC1 direction: ({eigenvectors[0,0]:.3f}, {eigenvectors[1,0]:.3f})")
print(f"PC2 direction: ({eigenvectors[0,1]:.3f}, {eigenvectors[1,1]:.3f})")
