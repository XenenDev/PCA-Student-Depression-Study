import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load NORMALIZED dataset for everything
# ============================================================
df = pd.read_csv('student_depression_normalized.csv')
cols = [c for c in df.columns if c != 'Depression']
data = df[cols].dropna()

print(f"Using NORMALIZED data")
print(f"Variables: {cols}")
print(f"Number of variables: {len(cols)}")
print(f"Sample size: {len(data)}")

# Compute covariance matrix on NORMALIZED data
cov_matrix = np.cov(data.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = np.argsort(eigenvalues.real)[::-1]
eigenvalues = eigenvalues.real[idx]
eigenvectors = eigenvectors.real[:, idx]

n_components = len(cols)

# ============================================================
# 1. PVE Chart (Proportion of Variance Explained)
# ============================================================
var_explained = eigenvalues / np.sum(eigenvalues) * 100
cumulative_var = np.cumsum(var_explained)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot - Individual PVE
ax1 = axes[0]
bars = ax1.bar(range(1, n_components+1), var_explained, color='steelblue', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Variance Explained (%)', fontsize=12)
ax1.set_title('Proportion of Variance Explained (PVE) by Each PC', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, n_components+1))
ax1.set_xticklabels([f'PC{i}' for i in range(1, n_components+1)], rotation=45, ha='right')
ax1.axhline(y=100/n_components, color='red', linestyle='--', label=f'Equal variance ({100/n_components:.1f}%)')
ax1.legend()
for i, v in enumerate(var_explained):
    ax1.text(i+1, v+0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# Right plot - Cumulative PVE
ax2 = axes[1]
ax2.plot(range(1, n_components+1), cumulative_var, 'o-', color='darkgreen', linewidth=2, markersize=8)
ax2.fill_between(range(1, n_components+1), cumulative_var, alpha=0.3, color='green')
ax2.set_xlabel('Number of Principal Components', fontsize=12)
ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, n_components+1))
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% threshold')
ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
ax2.legend()
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.3)
for i, v in enumerate(cumulative_var):
    ax2.annotate(f'{v:.1f}%', (i+1, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('pve_mixed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pve_mixed.png")

# ============================================================
# 2. Covariance Matrix Heatmap (on NORMALIZED data)
# ============================================================
plt.figure(figsize=(12, 10))
sns.heatmap(cov_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=cols, yticklabels=cols,
            vmin=-1, vmax=1, square=True)
plt.title('Covariance Matrix Heatmap\n(Normalized Data, Excluding Depression)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('covariance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: covariance_heatmap.png")

# ============================================================
# 3. PC Loadings Heatmap
# ============================================================
# Short variable names for display
short_names = ['Age', 'Acad. Press.', 'Work Press.', 'CGPA', 'Study Sat.', 
               'Job Sat.', 'Work/Study Hrs', 'Fin. Stress', 'Sleep', 
               'Suicidal', 'Fam. History', 'Is Female', 'Diet']

# Show first 10 PCs
n_show = 10
loadings_df = pd.DataFrame(eigenvectors[:, :n_show].T, 
                           columns=short_names,
                           index=[f'PC{i+1}' for i in range(n_show)])

plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Principal Component Loadings Heatmap\n(Weight of each variable in each PC)', fontsize=14, fontweight='bold')
plt.xlabel('Original Variables', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
plt.tight_layout()
plt.savefig('pc_loadings_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pc_loadings_heatmap.png")

# ============================================================
# 4. PC vs Depression Correlation Comparison
# ============================================================
# Project NORMALIZED data onto PCs
pc_scores = data.values @ eigenvectors

# Get depression values (aligned with data after dropna)
depression = df.loc[data.index, 'Depression'].values

# Correlations of PCs with Depression
pc_correlations = []
for i in range(n_components):
    corr = np.corrcoef(pc_scores[:, i], depression)[0, 1]
    pc_correlations.append(corr)

# Correlations of original NORMALIZED variables with Depression
var_correlations = []
for col in cols:
    corr = np.corrcoef(data[col].values, depression)[0, 1]
    var_correlations.append((col, corr))

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: PCs sorted by |correlation|
ax1 = axes[0]
pc_data = [(f'PC{i+1}', pc_correlations[i]) for i in range(n_components)]
pc_data_sorted = sorted(pc_data, key=lambda x: abs(x[1]))
pc_names = [x[0] for x in pc_data_sorted]
pc_corrs = [x[1] for x in pc_data_sorted]
colors1 = ['coral' if c > 0 else 'steelblue' for c in pc_corrs]
bars1 = ax1.barh(pc_names, pc_corrs, color=colors1, edgecolor='black', alpha=0.8)
ax1.set_xlabel('Pearson Correlation with Depression', fontsize=12)
ax1.set_title('PCs Sorted by |Correlation|', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.5)
for i, (name, corr) in enumerate(zip(pc_names, pc_corrs)):
    ax1.text(corr + 0.01 if corr >= 0 else corr - 0.01, i, f'{corr:.3f}', 
             va='center', ha='left' if corr >= 0 else 'right', fontsize=9)

# Right: Variables sorted by |correlation|
ax2 = axes[1]
var_data_sorted = sorted(var_correlations, key=lambda x: abs(x[1]))
var_names = [x[0] for x in var_data_sorted]
var_corrs = [x[1] for x in var_data_sorted]
colors2 = ['coral' if c > 0 else 'steelblue' for c in var_corrs]
bars2 = ax2.barh(var_names, var_corrs, color=colors2, edgecolor='black', alpha=0.8)
ax2.set_xlabel('Pearson Correlation with Depression', fontsize=12)
ax2.set_title('Variables Sorted by |Correlation|', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=0.5)
for i, (name, corr) in enumerate(zip(var_names, var_corrs)):
    ax2.text(corr + 0.01 if corr >= 0 else corr - 0.01, i, f'{corr:.3f}', 
             va='center', ha='left' if corr >= 0 else 'right', fontsize=9)

plt.suptitle('Comparison: PC vs Original Variable Correlations with Depression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pc_vs_variable_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pc_vs_variable_correlations.png")

# ============================================================
# 5. Eigenvalues and Eigenvectors Text File
# ============================================================
with open('eigenvalues_eigenvectors.txt', 'w') as f:
    f.write("EIGENVALUES AND EIGENVECTORS OF THE COVARIANCE MATRIX\n")
    f.write("(Computed from Z-score Normalized Data)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Variables (columns): {cols}\n\n")
    
    for i in range(n_components):
        f.write("=" * 60 + "\n")
        f.write(f"PRINCIPAL COMPONENT {i+1}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Eigenvalue: {eigenvalues[i]:.6f}\n")
        f.write(f"Variance explained: {var_explained[i]:.2f}%\n\n")
        f.write("Eigenvector:\n")
        for j, var in enumerate(cols):
            f.write(f"  {var:25s}: {eigenvectors[j, i]:.6f}\n")
        f.write("\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("SUMMARY TABLE\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'PC':<6}{'Eigenvalue':<16}{'Variance %':<13}{'Cumulative %'}\n")
    f.write("-" * 44 + "\n")
    for i in range(n_components):
        f.write(f"PC{i+1:<4}{eigenvalues[i]:<16.6f}{var_explained[i]:<13.2f}{cumulative_var[i]:.2f}\n")

print("Saved: eigenvalues_eigenvectors.txt")

# ============================================================
# 6. Scree Plot
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components+1), eigenvalues, 'bo-', linewidth=2, markersize=10)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.title('Scree Plot', fontsize=14, fontweight='bold')
plt.xticks(range(1, n_components+1), [f'PC{i}' for i in range(1, n_components+1)])
plt.axhline(y=1, color='red', linestyle='--', label='Kaiser criterion (eigenvalue = 1)')
plt.legend()
plt.grid(True, alpha=0.3)
for i, v in enumerate(eigenvalues):
    plt.annotate(f'{v:.2f}', (i+1, v), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('scree_plot_mixed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scree_plot_mixed.png")

# ============================================================
# Print Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY (All computed from NORMALIZED data)")
print("=" * 60)
print(f"\nVariables ({n_components}):")
for i, col in enumerate(cols):
    print(f"  {i+1}. {col}")

print(f"\nEigenvalues:")
for i, ev in enumerate(eigenvalues):
    print(f"  PC{i+1}: {ev:.4f} ({var_explained[i]:.2f}%)")

print(f"\nComponents with eigenvalue >= 1 (Kaiser criterion): {sum(eigenvalues >= 1)}")

# Find PC most correlated with depression
max_corr_idx = np.argmax(np.abs(pc_correlations))
print(f"\nPC most correlated with Depression: PC{max_corr_idx+1} (r = {pc_correlations[max_corr_idx]:.3f})")

print("\nAll files regenerated successfully using NORMALIZED data!")
