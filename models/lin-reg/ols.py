import numpy as np

class OLS:
    def __init__(self, X: np.array, y: np.array):
        self.X = X
        self.y = y

    def fit(self):
        """Fits the standard OLS model."""
        self.beta = np.linalg.inv(np.transpose(self.X) @ self.X) @ np.transpose(self.X) @ self.y

    def predict(self, X_test):
        """Predicts based on the linear coeff. beta"""
        print(self.beta @ X_test.values)
        return X_test @ self.beta