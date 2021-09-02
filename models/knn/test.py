import math
import pandas as pd
from .solution import predict
from ..utils.log import logger

def load_data():
    return pd.read_csv("./data/reg_train.csv"), pd.read_csv("./data/reg_test.csv")

def test_solution(train, test):
    assert(round(math.sqrt(((test["OUTCOME"] - predict(train, test, k=5))**2).mean()), 7) == 0.3638982)


if __name__ == "__main__":
    train, test = load_data()
    test_solution(train, test)
    logger.info("All tests passed.")