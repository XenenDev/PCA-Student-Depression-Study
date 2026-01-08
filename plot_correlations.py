import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load normalized data
df = pd.read_csv('student_depression_normalized.csv')
cols = [c for c in df.columns if c != 'Depression']
data = df[cols].dropna()

# Compute PCA
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues.real[idx]
eigenvectors = eigenvectors.real[:, idx]
n_components = len(cols)

# Project data onto PCs
pc_scores = data.values @ eigenvectors
depression = df.loc[data.index, 'Depression'].values

# Correlations of PCs with Depression
pc_correlations = [np.corrcoef(pc_scores[:, i], depression)[0, 1] for i in range(n_components)]

# Correlations of original variables with Depression
var_correlations = [(col, np.corrcoef(data[col].values, depression)[0, 1]) for col in cols]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# Left: PCs sorted by |correlation| (highest at top)
ax1 = axes[0]
pc_data = [(f'PC{i+1}', pc_correlations[i]) for i in range(n_components)]
pc_data_sorted = sorted(pc_data, key=lambda x: abs(x[1]), reverse=True)
pc_names = [x[0] for x in pc_data_sorted]
pc_corrs = [x[1] for x in pc_data_sorted]
colors1 = ['coral' if c > 0 else 'steelblue' for c in pc_corrs]

bars1 = ax1.barh(range(len(pc_names)), pc_corrs, color=colors1, edgecolor='black', alpha=0.8)
ax1.set_yticks(range(len(pc_names)))
ax1.set_yticklabels(pc_names, fontsize=11)
ax1.set_xlabel('Pearson Correlation with Depression', fontsize=12)
ax1.set_title('PCs Sorted by |Correlation|', fontsize=13, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.5)
ax1.set_xlim(-0.8, 0.15)
ax1.invert_yaxis()

# Add value labels for left plot
for i, (name, corr) in enumerate(zip(pc_names, pc_corrs)):
    # Position label inside or outside bar depending on value
    if corr < -0.1:
        ax1.text(corr + 0.02, i, f'{corr:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
    elif corr < 0:
        ax1.text(corr - 0.02, i, f'{corr:.3f}', va='center', ha='right', fontsize=9)
    else:
        ax1.text(corr + 0.02, i, f'{corr:.3f}', va='center', ha='left', fontsize=9)

# Right: Variables sorted by |correlation| (highest at top)
ax2 = axes[1]
var_data_sorted = sorted(var_correlations, key=lambda x: abs(x[1]), reverse=True)
var_names = [x[0] for x in var_data_sorted]
var_corrs = [x[1] for x in var_data_sorted]
colors2 = ['coral' if c > 0 else 'steelblue' for c in var_corrs]

bars2 = ax2.barh(range(len(var_names)), var_corrs, color=colors2, edgecolor='black', alpha=0.8)
ax2.set_yticks(range(len(var_names)))
ax2.set_yticklabels(var_names, fontsize=10)
ax2.set_xlabel('Pearson Correlation with Depression', fontsize=12)
ax2.set_title('Variables Sorted by |Correlation|', fontsize=13, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=0.5)
ax2.set_xlim(-0.35, 0.65)
ax2.invert_yaxis()

# Add value labels for right plot
for i, (name, corr) in enumerate(zip(var_names, var_corrs)):
    if corr > 0.1:
        ax2.text(corr + 0.02, i, f'{corr:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
    elif corr > 0:
        ax2.text(corr + 0.02, i, f'{corr:.3f}', va='center', ha='left', fontsize=9)
    elif corr > -0.1:
        ax2.text(corr - 0.02, i, f'{corr:.3f}', va='center', ha='right', fontsize=9)
    else:
        ax2.text(corr - 0.02, i, f'{corr:.3f}', va='center', ha='right', fontsize=9)

plt.suptitle('Comparison: PC vs Original Variable Correlations with Depression', 
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pc_vs_variable_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: pc_vs_variable_correlations.png')
