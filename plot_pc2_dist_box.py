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

# Split by depression status
pc2_not_depressed = pc2[depression == 0]
pc2_depressed = pc2[depression == 1]

# Statistics
mean_nd = np.mean(pc2_not_depressed)
mean_d = np.mean(pc2_depressed)
std_nd = np.std(pc2_not_depressed)
std_d = np.std(pc2_depressed)

# t-test
t_stat, p_value = stats.ttest_ind(pc2_depressed, pc2_not_depressed, equal_var=False)

# Cohen's d
n1, n2 = len(pc2_not_depressed), len(pc2_depressed)
s1, s2 = np.std(pc2_not_depressed, ddof=1), np.std(pc2_depressed, ddof=1)
pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
cohens_d = (mean_d - mean_nd) / pooled_std

# ============================================================
# FIGURE 1: Normal Distribution Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram bins - tighter range around the actual data
x_min = -5
x_max = 5
bins = np.linspace(x_min, x_max, 50)

# Plot histograms
ax.hist(pc2_not_depressed, bins=bins, alpha=0.5, color='steelblue', 
        density=True, edgecolor='white', linewidth=0.5,
        label=f'Not Depressed (n={len(pc2_not_depressed):,})')
ax.hist(pc2_depressed, bins=bins, alpha=0.5, color='coral', 
        density=True, edgecolor='white', linewidth=0.5,
        label=f'Depressed (n={len(pc2_depressed):,})')

# Overlay fitted normal curves
x_curve = np.linspace(x_min, x_max, 500)
y_nd = stats.norm.pdf(x_curve, mean_nd, std_nd)
y_d = stats.norm.pdf(x_curve, mean_d, std_d)

ax.plot(x_curve, y_nd, 'darkblue', lw=2.5, label=f'Normal fit: μ={mean_nd:.2f}, σ={std_nd:.2f}')
ax.plot(x_curve, y_d, 'darkred', lw=2.5, label=f'Normal fit: μ={mean_d:.2f}, σ={std_d:.2f}')

# Add vertical lines for means
ax.axvline(mean_nd, color='darkblue', linestyle='--', lw=2, alpha=0.8)
ax.axvline(mean_d, color='darkred', linestyle='--', lw=2, alpha=0.8)

# Labels and formatting
ax.set_xlabel('PC2 Score', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Distribution of PC2 Scores by Depression Status', fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pc2_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pc2_distribution.png")

# ============================================================
# FIGURE 2: Box Plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

bp = ax.boxplot([pc2_not_depressed, pc2_depressed], 
                tick_labels=['Not Depressed', 'Depressed'],
                patch_artist=True,
                widths=0.5,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='yellow', 
                              markeredgecolor='black', markersize=8))

colors = ['steelblue', 'coral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add significance bracket
y_max = max(pc2.max(), 4.5)
ax.plot([1, 1, 2, 2], [y_max+0.2, y_max+0.4, y_max+0.4, y_max+0.2], 'k-', lw=1.5)
ax.text(1.5, y_max+0.5, '***\np < 0.001', ha='center', fontsize=12, fontweight='bold')

# Labels
ax.set_ylabel('PC2 Score', fontsize=14)
ax.set_title('PC2 Score by Depression Status', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add mean values as text
ax.text(1, mean_nd + 0.15, f'μ = {mean_nd:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.text(2, mean_d - 0.25, f'μ = {mean_d:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('pc2_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pc2_boxplot.png")
