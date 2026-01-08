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

# Extract PC2 scores
pc2 = pc_scores[:, 1]

# Fit logistic regression: Depression ~ PC2
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_loss(params, X, y):
    beta0, beta1 = params
    p = sigmoid(beta0 + beta1 * X)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# Fit model
result = minimize(logistic_loss, [0, 0], args=(pc2, depression), method='BFGS')
beta0, beta1 = result.x

print(f"Logistic Regression: Depression ~ PC2")
print(f"  Intercept (β₀): {beta0:.4f}")
print(f"  Slope (β₁): {beta1:.4f}")
print(f"  Model: P(Depression=1) = 1 / (1 + exp(-({beta0:.3f} + {beta1:.3f}×PC2)))")

# Calculate correlation
corr = np.corrcoef(pc2, depression)[0, 1]
print(f"  Pearson correlation: r = {corr:.4f}")

# Create figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ============================================================
# Plot 1: PC2 Distribution by Depression Status
# ============================================================
ax = axes[0, 0]
pc2_dep0 = pc2[depression == 0]
pc2_dep1 = pc2[depression == 1]

ax.hist(pc2_dep0, bins=50, alpha=0.6, color='steelblue', label=f'Not Depressed (n={len(pc2_dep0)})', density=True)
ax.hist(pc2_dep1, bins=50, alpha=0.6, color='coral', label=f'Depressed (n={len(pc2_dep1)})', density=True)
ax.axvline(np.mean(pc2_dep0), color='steelblue', linestyle='--', lw=2, label=f'Mean (Not Dep): {np.mean(pc2_dep0):.2f}')
ax.axvline(np.mean(pc2_dep1), color='coral', linestyle='--', lw=2, label=f'Mean (Dep): {np.mean(pc2_dep1):.2f}')
ax.set_xlabel('PC2 Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of PC2 Scores by Depression Status', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ============================================================
# Plot 2: Logistic Regression Curve
# ============================================================
ax = axes[0, 1]

# Scatter with jitter on y-axis for visibility
jitter = np.random.uniform(-0.05, 0.05, len(depression))
ax.scatter(pc2, depression + jitter, alpha=0.1, s=5, c='gray')

# Plot logistic curve
x_range = np.linspace(pc2.min(), pc2.max(), 500)
y_pred = sigmoid(beta0 + beta1 * x_range)
ax.plot(x_range, y_pred, 'r-', lw=3, label=f'Logistic fit: P(Dep) = σ({beta0:.2f} + {beta1:.2f}×PC2)')

# Add decision boundary
decision_boundary = -beta0 / beta1
ax.axvline(decision_boundary, color='green', linestyle='--', lw=2, 
           label=f'Decision boundary: PC2 = {decision_boundary:.2f}')
ax.axhline(0.5, color='orange', linestyle=':', alpha=0.7)

ax.set_xlabel('PC2 Score', fontsize=12)
ax.set_ylabel('Depression (0 or 1)', fontsize=12)
ax.set_title('Logistic Regression: Depression ~ PC2', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(-0.15, 1.15)
ax.set_yticks([0, 0.5, 1])
ax.grid(True, alpha=0.3)

# ============================================================
# Plot 3: Binned Probability Plot
# ============================================================
ax = axes[1, 0]

# Bin PC2 scores and calculate actual depression rate in each bin
n_bins = 20
bins = np.linspace(pc2.min(), pc2.max(), n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_probs = []
bin_counts = []

for i in range(n_bins):
    mask = (pc2 >= bins[i]) & (pc2 < bins[i+1])
    if mask.sum() > 0:
        bin_probs.append(depression[mask].mean())
        bin_counts.append(mask.sum())
    else:
        bin_probs.append(np.nan)
        bin_counts.append(0)

# Plot actual binned probabilities
ax.bar(bin_centers, bin_probs, width=(bins[1]-bins[0])*0.8, alpha=0.6, 
       color='steelblue', edgecolor='black', label='Observed rate')

# Overlay logistic prediction
ax.plot(x_range, y_pred, 'r-', lw=3, label='Logistic model prediction')

ax.set_xlabel('PC2 Score (binned)', fontsize=12)
ax.set_ylabel('P(Depression = 1)', fontsize=12)
ax.set_title('Observed vs Predicted Depression Rate by PC2 Bin', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

# ============================================================
# Plot 4: ROC Curve
# ============================================================
ax = axes[1, 1]

# Calculate predicted probabilities
y_prob = sigmoid(beta0 + beta1 * pc2)

# Calculate ROC curve
thresholds = np.linspace(0, 1, 200)
tpr_list = []
fpr_list = []

for thresh in thresholds:
    y_pred_binary = (y_prob >= thresh).astype(int)
    tp = np.sum((y_pred_binary == 1) & (depression == 1))
    fp = np.sum((y_pred_binary == 1) & (depression == 0))
    tn = np.sum((y_pred_binary == 0) & (depression == 0))
    fn = np.sum((y_pred_binary == 0) & (depression == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_list.append(tpr)
    fpr_list.append(fpr)

# Calculate AUC (simple trapezoidal)
sorted_idx = np.argsort(fpr_list)
fpr_sorted = np.array(fpr_list)[sorted_idx]
tpr_sorted = np.array(tpr_list)[sorted_idx]
auc = np.trapz(tpr_sorted, fpr_sorted)

ax.plot(fpr_list, tpr_list, 'b-', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.fill_between(fpr_sorted, tpr_sorted, alpha=0.2, color='blue')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve: PC2 as Depression Predictor', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

plt.suptitle('PC2 as a Predictor of Depression: Logistic Regression Analysis', 
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('pc2_regression_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pc2_regression_analysis.png")

# Print model accuracy
y_pred_binary = (y_prob >= 0.5).astype(int)
accuracy = np.mean(y_pred_binary == depression)
print(f"\nModel accuracy (threshold=0.5): {accuracy:.1%}")
print(f"AUC: {auc:.3f}")
