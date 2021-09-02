import numpy as np
import pandas as pd
from ols import OLS
# from ..utils.log import logger

def load_data():
    train = pd.read_csv("data/reg_train.csv")
    test = pd.read_csv("data/reg_test.csv")
    X_train, X_test = train.drop("OUTCOME", axis=1), test.drop("OUTCOME", axis=1)
    y_train, y_test = train["OUTCOME"], test["OUTCOME"]
    return X_train, y_train, X_test, y_test 

def mse(y, y_hat):
    """Calculates the mean squared error."""
    _errors = y - y_hat
    _n = len(y)
    return (np.transpose(_errors) @ _errors)/_n

def test_ols(X_train, y_train, X_test, y_test):
    ols = OLS(X_train, y_train)
    ols.fit()
    y_hat = ols.predict(X_test)
    print("Mean Squared Error:", mse(y_test, y_hat))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    test_ols(X_train, y_train, X_test, y_test)