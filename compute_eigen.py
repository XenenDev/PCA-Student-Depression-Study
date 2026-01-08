import pandas as pd
import numpy as np

# Load the normalized dataset
df = pd.read_csv('student_depression_normalized.csv')

# Get all columns except Depression
cols = [c for c in df.columns if c != 'Depression']
data = df[cols].dropna()

# Compute covariance matrix
cov_matrix = np.cov(data.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Save to file
with open('eigenvalues_eigenvectors.txt', 'w') as f:
    f.write("EIGENVALUES AND EIGENVECTORS OF THE COVARIANCE MATRIX\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Variables (columns): {cols}\n\n")
    
    for i in range(len(eigenvalues)):
        f.write(f"{'=' * 60}\n")
        f.write(f"PRINCIPAL COMPONENT {i + 1}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Eigenvalue: {eigenvalues[i]:.6f}\n")
        f.write(f"Variance explained: {(eigenvalues[i] / np.sum(eigenvalues) * 100):.2f}%\n\n")
        f.write("Eigenvector:\n")
        for j, var in enumerate(cols):
            f.write(f"  {var:25s}: {eigenvectors[j, i]:.6f}\n")
        f.write("\n")
    
    # Summary table
    f.write("\n" + "=" * 60 + "\n")
    f.write("SUMMARY TABLE\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'PC':<5} {'Eigenvalue':<15} {'Variance %':<12} {'Cumulative %':<12}\n")
    f.write("-" * 44 + "\n")
    cumulative = 0
    for i in range(len(eigenvalues)):
        var_pct = eigenvalues[i] / np.sum(eigenvalues) * 100
        cumulative += var_pct
        f.write(f"PC{i+1:<3} {eigenvalues[i]:<15.6f} {var_pct:<12.2f} {cumulative:<12.2f}\n")

print("Results saved to 'eigenvalues_eigenvectors.txt'")
print("\n" + "=" * 60)
print("EIGENVALUES (sorted by magnitude):")
print("=" * 60)
for i, ev in enumerate(eigenvalues):
    print(f"PC{i+1}: {ev:.6f} ({ev/np.sum(eigenvalues)*100:.2f}% variance)")

print("\n" + "=" * 60)
print("EIGENVECTORS (as columns):")
print("=" * 60)
print("\nVariables:")
for i, col in enumerate(cols):
    print(f"  {i+1}. {col}")
print("\nEigenvector matrix (each column is an eigenvector):")
np.set_printoptions(precision=4, linewidth=200, suppress=True)
print(eigenvectors)
