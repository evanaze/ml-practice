import numpy as np

class OLS:
    def __init__(self, X: np.array, y: np.array):
        self.X = X
        self.y = y

    def fit(self):
        """Fits the standard OLS model."""
        self.beta = np.linalg.inv(self.X @ np.transpose(self.X)) @ np.transpose(self.X) @ self.y

    def predict(self):
        """Predicts based on the linear coeff. beta"""
        return self.X @ self.beta 