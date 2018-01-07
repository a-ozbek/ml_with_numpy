import numpy as np


class PCA:
    """
    Principal Component Analysis
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.eigvalues = None
        self.eigvectors = None
    
    def fit(self, X):
        cov_matrix = np.cov(X.T)
        self.eigvalues, self.eigvectors = np.linalg.eig(cov_matrix)
        # Sort eigen vectors w.r.t eigenvalues
        eig_sorting = np.argsort(self.eigvalues)[::-1]  # Descending
        self.eigvalues = self.eigvalues[eig_sorting]
        self.eigvectors = self.eigvectors[:, eig_sorting]        
    
    def transform(self, X):
        X_reduced = np.matrix(X) * np.matrix(self.eigvectors[:, :self.n_components])
        return np.array(X_reduced)
