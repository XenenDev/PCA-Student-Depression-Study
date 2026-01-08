import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Create figure with 6 scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Color map for depression (0 = blue, 1 = red)
colors = ['steelblue' if d == 0 else 'coral' for d in depression]
alpha = 0.3

# 1. PC1 vs PC2 (top 2 components)
ax = axes[0, 0]
ax.scatter(pc_scores[:, 0], pc_scores[:, 1], c=colors, alpha=alpha, s=10)
ax.set_xlabel('PC1', fontsize=11)
ax.set_ylabel('PC2', fontsize=11)
ax.set_title('PC1 vs PC2\n(Top 2 Principal Components)', fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

# 2. PC2 vs PC3 (PC2 is most correlated with depression)
ax = axes[0, 1]
ax.scatter(pc_scores[:, 1], pc_scores[:, 2], c=colors, alpha=alpha, s=10)
ax.set_xlabel('PC2', fontsize=11)
ax.set_ylabel('PC3', fontsize=11)
ax.set_title('PC2 vs PC3\n(PC2 = Strongest Depression Predictor)', fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

# 3. Academic Pressure vs Suicidal Thoughts
ax = axes[0, 2]
# Add jitter to see density better (these are discrete-ish values)
jitter = 0.1
x = data['Academic Pressure'].values + np.random.normal(0, jitter, len(data))
y = data['Suicidal Thoughts'].values + np.random.normal(0, jitter, len(data))
ax.scatter(x, y, c=colors, alpha=alpha, s=10)
ax.set_xlabel('Academic Pressure (normalized)', fontsize=11)
ax.set_ylabel('Suicidal Thoughts (normalized)', fontsize=11)
ax.set_title('Academic Pressure vs Suicidal Thoughts', fontsize=12, fontweight='bold')

# 4. Financial Stress vs Sleep Duration
ax = axes[1, 0]
x = data['Financial Stress'].values + np.random.normal(0, jitter, len(data))
y = data['Sleep Duration'].values + np.random.normal(0, jitter, len(data))
ax.scatter(x, y, c=colors, alpha=alpha, s=10)
ax.set_xlabel('Financial Stress (normalized)', fontsize=11)
ax.set_ylabel('Sleep Duration (normalized)', fontsize=11)
ax.set_title('Financial Stress vs Sleep Duration', fontsize=12, fontweight='bold')

# 5. CGPA vs Study Satisfaction
ax = axes[1, 1]
x = data['CGPA'].values + np.random.normal(0, jitter, len(data))
y = data['Study Satisfaction'].values + np.random.normal(0, jitter, len(data))
ax.scatter(x, y, c=colors, alpha=alpha, s=10)
ax.set_xlabel('CGPA (normalized)', fontsize=11)
ax.set_ylabel('Study Satisfaction (normalized)', fontsize=11)
ax.set_title('CGPA vs Study Satisfaction', fontsize=12, fontweight='bold')

# 6. Work Pressure vs Job Satisfaction (the PC1 dominant pair)
ax = axes[1, 2]
x = data['Work Pressure'].values + np.random.normal(0, jitter, len(data))
y = data['Job Satisfaction'].values + np.random.normal(0, jitter, len(data))
ax.scatter(x, y, c=colors, alpha=alpha, s=10)
ax.set_xlabel('Work Pressure (normalized)', fontsize=11)
ax.set_ylabel('Job Satisfaction (normalized)', fontsize=11)
ax.set_title('Work Pressure vs Job Satisfaction\n(Dominant in PC1)', fontsize=12, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Not Depressed'),
                   Patch(facecolor='coral', alpha=0.7, label='Depressed')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, 
           bbox_to_anchor=(0.5, 0.98))

plt.suptitle('Scatter Plot Suite: PCA Space & Key Variable Relationships', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('scatter_suite.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: scatter_suite.png")
