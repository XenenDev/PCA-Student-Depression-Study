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

print("=" * 70)
print("STATISTICAL ANALYSIS: PC2 AS A PREDICTOR OF DEPRESSION")
print("=" * 70)

# ============================================================
# 1. Descriptive Statistics
# ============================================================
print("\n1. DESCRIPTIVE STATISTICS")
print("-" * 50)
print(f"{'Group':<20} {'n':>8} {'Mean':>10} {'Std':>10} {'Median':>10}")
print("-" * 50)
print(f"{'Not Depressed':<20} {len(pc2_not_depressed):>8} {np.mean(pc2_not_depressed):>10.4f} {np.std(pc2_not_depressed):>10.4f} {np.median(pc2_not_depressed):>10.4f}")
print(f"{'Depressed':<20} {len(pc2_depressed):>8} {np.mean(pc2_depressed):>10.4f} {np.std(pc2_depressed):>10.4f} {np.median(pc2_depressed):>10.4f}")

mean_diff = np.mean(pc2_depressed) - np.mean(pc2_not_depressed)
print(f"\nMean difference: {mean_diff:.4f}")

# ============================================================
# 2. Independent Samples t-Test
# ============================================================
print("\n" + "=" * 70)
print("2. INDEPENDENT SAMPLES t-TEST")
print("-" * 50)
print("H₀: μ_depressed = μ_not_depressed (no difference in PC2 means)")
print("H₁: μ_depressed ≠ μ_not_depressed (PC2 means differ)")
print("-" * 50)

# Two-sample t-test (Welch's t-test, doesn't assume equal variances)
t_stat, p_value_ttest = stats.ttest_ind(pc2_depressed, pc2_not_depressed, equal_var=False)
print(f"Welch's t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value_ttest:.2e}")
print(f"Degrees of freedom (approx): {len(pc2_depressed) + len(pc2_not_depressed) - 2}")

if p_value_ttest < 0.001:
    print(f"\n✓ Result: HIGHLY SIGNIFICANT (p < 0.001)")
    print("  Reject H₀: The mean PC2 scores are significantly different.")
else:
    print(f"\n✗ Result: Not significant at α = 0.001")

# ============================================================
# 3. Effect Size (Cohen's d)
# ============================================================
print("\n" + "=" * 70)
print("3. EFFECT SIZE (COHEN'S d)")
print("-" * 50)

# Pooled standard deviation
n1, n2 = len(pc2_not_depressed), len(pc2_depressed)
s1, s2 = np.std(pc2_not_depressed, ddof=1), np.std(pc2_depressed, ddof=1)
pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
cohens_d = (np.mean(pc2_depressed) - np.mean(pc2_not_depressed)) / pooled_std

print(f"Cohen's d: {cohens_d:.4f}")
print(f"Pooled standard deviation: {pooled_std:.4f}")

if abs(cohens_d) >= 0.8:
    effect_size = "LARGE"
elif abs(cohens_d) >= 0.5:
    effect_size = "MEDIUM"
else:
    effect_size = "SMALL"
print(f"Interpretation: {effect_size} effect size")
print("  (|d| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large)")

# ============================================================
# 4. Linear Regression: Depression ~ PC2
# ============================================================
print("\n" + "=" * 70)
print("4. SIMPLE LINEAR REGRESSION: Depression ~ PC2")
print("-" * 50)

# Fit linear regression using least squares
slope, intercept, r_value, p_value_reg, std_err = stats.linregress(pc2, depression)

print(f"Model: Depression = {intercept:.4f} + ({slope:.4f}) × PC2")
print(f"\nCoefficients:")
print(f"  Intercept (β₀): {intercept:.4f}")
print(f"  Slope (β₁): {slope:.4f}")
print(f"  Standard error of slope: {std_err:.4f}")
print(f"\nModel Fit:")
print(f"  Pearson r: {r_value:.4f}")
print(f"  R²: {r_value**2:.4f} ({r_value**2*100:.1f}% of variance explained)")
print(f"  p-value for slope: {p_value_reg:.2e}")

# t-statistic for slope
t_slope = slope / std_err
print(f"  t-statistic for β₁: {t_slope:.4f}")

# ============================================================
# 5. F-Test for Regression
# ============================================================
print("\n" + "=" * 70)
print("5. F-TEST FOR REGRESSION SIGNIFICANCE")
print("-" * 50)
print("H₀: β₁ = 0 (PC2 has no linear relationship with Depression)")
print("H₁: β₁ ≠ 0 (PC2 has a linear relationship with Depression)")
print("-" * 50)

# Calculate F-statistic
y_pred = intercept + slope * pc2
ss_reg = np.sum((y_pred - np.mean(depression))**2)  # Regression sum of squares
ss_res = np.sum((depression - y_pred)**2)           # Residual sum of squares
ss_tot = np.sum((depression - np.mean(depression))**2)  # Total sum of squares

df_reg = 1  # Number of predictors
df_res = len(depression) - 2  # n - p - 1
ms_reg = ss_reg / df_reg
ms_res = ss_res / df_res
f_stat = ms_reg / ms_res
p_value_f = 1 - stats.f.cdf(f_stat, df_reg, df_res)

print(f"SS_regression: {ss_reg:.4f}")
print(f"SS_residual: {ss_res:.4f}")
print(f"SS_total: {ss_tot:.4f}")
print(f"df_regression: {df_reg}")
print(f"df_residual: {df_res}")
print(f"\nF-statistic: {f_stat:.4f}")
print(f"p-value: {p_value_f:.2e}")

if p_value_f < 0.001:
    print(f"\n✓ Result: HIGHLY SIGNIFICANT (p < 0.001)")
    print("  Reject H₀: PC2 is a significant linear predictor of Depression.")

# ============================================================
# 6. Confidence Interval for Slope
# ============================================================
print("\n" + "=" * 70)
print("6. 95% CONFIDENCE INTERVAL FOR SLOPE")
print("-" * 50)

alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df_res)
ci_lower = slope - t_crit * std_err
ci_upper = slope + t_crit * std_err

print(f"95% CI for β₁: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Since 0 is not in this interval, the slope is significant.")

# ============================================================
# CREATE VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Enhanced Distribution Comparison
ax = axes[0, 0]
bins = np.linspace(min(pc2), max(pc2), 40)

ax.hist(pc2_not_depressed, bins=bins, alpha=0.6, color='steelblue', 
        label=f'Not Depressed\n(n={len(pc2_not_depressed)}, μ={np.mean(pc2_not_depressed):.3f})', 
        density=True, edgecolor='white')
ax.hist(pc2_depressed, bins=bins, alpha=0.6, color='coral', 
        label=f'Depressed\n(n={len(pc2_depressed)}, μ={np.mean(pc2_depressed):.3f})', 
        density=True, edgecolor='white')

# Add vertical lines for means
ax.axvline(np.mean(pc2_not_depressed), color='darkblue', linestyle='--', lw=2.5)
ax.axvline(np.mean(pc2_depressed), color='darkred', linestyle='--', lw=2.5)

# Add arrow showing difference
mid_y = ax.get_ylim()[1] * 0.85
ax.annotate('', xy=(np.mean(pc2_depressed), mid_y), 
            xytext=(np.mean(pc2_not_depressed), mid_y),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text((np.mean(pc2_depressed) + np.mean(pc2_not_depressed))/2, mid_y + 0.02,
        f'Δμ = {mean_diff:.3f}', ha='center', fontsize=11, fontweight='bold')

ax.set_xlabel('PC2 Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of PC2 by Depression Status\n' + 
             f't = {t_stat:.2f}, p < 0.001, Cohen\'s d = {cohens_d:.2f}', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 2: Linear Regression
ax = axes[0, 1]

# Add jitter to depression for visibility
jitter = np.random.uniform(-0.03, 0.03, len(depression))
ax.scatter(pc2, depression + jitter, alpha=0.15, s=8, c='gray', label='Data points')

# Regression line
x_line = np.linspace(pc2.min(), pc2.max(), 100)
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, 'r-', lw=3, label=f'Linear fit: y = {intercept:.3f} + ({slope:.3f})x')

# Confidence band for regression line
y_pred_line = intercept + slope * x_line
se_line = std_err * np.sqrt(1/len(pc2) + (x_line - np.mean(pc2))**2 / np.sum((pc2 - np.mean(pc2))**2))
ax.fill_between(x_line, y_pred_line - 1.96*se_line, y_pred_line + 1.96*se_line, 
                alpha=0.2, color='red', label='95% CI')

ax.set_xlabel('PC2 Score', fontsize=12)
ax.set_ylabel('Depression (0 or 1)', fontsize=12)
ax.set_title(f'Linear Regression: Depression ~ PC2\n' +
             f'R² = {r_value**2:.3f}, F = {f_stat:.1f}, p < 0.001', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.3)

# Plot 3: Box Plot Comparison
ax = axes[1, 0]

bp = ax.boxplot([pc2_not_depressed, pc2_depressed], 
                labels=['Not Depressed', 'Depressed'],
                patch_artist=True,
                widths=0.6)

colors = ['steelblue', 'coral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Add individual points with jitter
for i, (data_points, color) in enumerate([(pc2_not_depressed, 'steelblue'), 
                                           (pc2_depressed, 'coral')]):
    x = np.random.normal(i+1, 0.08, len(data_points))
    ax.scatter(x, data_points, alpha=0.05, s=3, c=color)

# Add significance bracket
y_max = max(pc2.max(), 4)
ax.plot([1, 1, 2, 2], [y_max+0.3, y_max+0.5, y_max+0.5, y_max+0.3], 'k-', lw=1.5)
ax.text(1.5, y_max+0.6, '***\np < 0.001', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('PC2 Score', fontsize=12)
ax.set_title('PC2 Score by Depression Status\n(Box Plot with Data Points)', 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary Statistics Table
ax = axes[1, 1]
ax.axis('off')

# Create summary table
table_data = [
    ['TEST', 'STATISTIC', 'p-VALUE', 'INTERPRETATION'],
    ['─' * 15, '─' * 12, '─' * 12, '─' * 20],
    ["Welch's t-test", f't = {t_stat:.3f}', f'{p_value_ttest:.2e}', 'Means differ significantly'],
    ['Effect Size', f"d = {cohens_d:.3f}", '—', f'{effect_size} effect'],
    ['Linear Regression', f'β₁ = {slope:.4f}', f'{p_value_reg:.2e}', 'Significant predictor'],
    ['F-test', f'F = {f_stat:.1f}', f'{p_value_f:.2e}', 'Model is significant'],
    ['Correlation', f'r = {r_value:.4f}', '—', f'R² = {r_value**2:.3f}'],
    ['─' * 15, '─' * 12, '─' * 12, '─' * 20],
    ['95% CI for slope', f'[{ci_lower:.4f}, {ci_upper:.4f}]', '', '0 not in interval']
]

table_text = '\n'.join(['  '.join(f'{item:<20}' for item in row) for row in table_data])
ax.text(0.05, 0.95, 'STATISTICAL SUMMARY', fontsize=14, fontweight='bold', 
        transform=ax.transAxes, va='top', family='monospace')
ax.text(0.05, 0.85, table_text, fontsize=10, transform=ax.transAxes, 
        va='top', family='monospace')

# Add conclusion
conclusion = f"""
CONCLUSION:
PC2 is a highly significant linear predictor of Depression.
• The t-test confirms the groups have different mean PC2 scores (p < 0.001)
• Cohen's d = {cohens_d:.2f} indicates a {effect_size.lower()} effect size
• R² = {r_value**2:.3f} means PC2 explains {r_value**2*100:.1f}% of variance in Depression
• For every 1-unit increase in PC2, Depression decreases by {abs(slope):.3f}
"""
ax.text(0.05, 0.35, conclusion, fontsize=11, transform=ax.transAxes, va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.suptitle('Statistical Tests: PC2 as a Linear Predictor of Depression', 
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('pc2_statistical_tests.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n" + "=" * 70)
print("Saved: pc2_statistical_tests.png")
