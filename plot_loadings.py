import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('student_depression_normalized.csv')
cols = [c for c in df.columns if c != 'Depression']
data = df[cols].dropna()

# Compute PCA
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues.real)[::-1]
eigenvectors = eigenvectors.real[:, idx]

# Variable names
short_names = ['Age', 'Acad. Press.', 'Work Press.', 'CGPA', 'Study Sat.', 
               'Job Sat.', 'Work/Study Hrs', 'Fin. Stress', 'Sleep', 
               'Suicidal', 'Fam. History', 'Is Female', 'Diet']

# Create loadings dataframe (13 PCs x 13 variables)
n_show = 13
loadings_df = pd.DataFrame(eigenvectors[:, :n_show].T, 
                           columns=short_names,
                           index=[f'PC{i+1}' for i in range(n_show)])

# Create figure with custom colorbar size
fig, ax = plt.subplots(figsize=(14, 10))

# Create heatmap with colorbar matching grid height
heatmap = sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                      vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax,
                      cbar_kws={'shrink': 0.77})

plt.title('Principal Component Loadings Heatmap\n(Weight of each variable in each PC)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Original Variables', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)

plt.tight_layout()
plt.savefig('pc_loadings_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: pc_loadings_heatmap.png')
