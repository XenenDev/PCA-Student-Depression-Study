import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load normalized data
df = pd.read_csv('student_depression_normalized.csv')
cols = [c for c in df.columns if c != 'Depression']
data = df[cols].dropna()
depression = df.loc[data.index, 'Depression'].values

# Compute PCA
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues.real[idx]
eigenvectors = eigenvectors.real[:, idx]

# Project data onto PCs
pc_scores = data.values @ eigenvectors
pc2 = pc_scores[:, 1]

# Fit linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(pc2, depression)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot with jitter on y-axis for visibility
jitter = np.random.uniform(-0.08, 0.08, len(depression))
ax.scatter(pc2, depression + jitter, alpha=0.15, s=8, c='steelblue', edgecolors='none')

# Best fit line
x_line = np.linspace(pc2.min(), pc2.max(), 100)
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, 'r-', lw=3, label=f'Best fit: y = {intercept:.3f} + ({slope:.3f})x')

# Labels and formatting
ax.set_xlabel('PC2 Score', fontsize=14)
ax.set_ylabel('Depression (0 or 1)', fontsize=14)
ax.set_title('Linear Regression: Depression vs PC2', fontsize=16, fontweight='bold')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Not Depressed (0)', 'Depressed (1)'])
ax.set_xlim(-5, 5)
ax.set_ylim(-0.2, 1.2)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)

# Add R² annotation
ax.text(0.02, 0.97, f'R² = {r_value**2:.3f}', transform=ax.transAxes, fontsize=13,
        va='top', ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig('pc2_linear_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pc2_linear_regression.png")
print(f"\nLinear model: Depression = {intercept:.4f} + ({slope:.4f}) × PC2")
print(f"R² = {r_value**2:.4f}")
