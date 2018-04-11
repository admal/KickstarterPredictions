import xgboost as xgb
from time import time
import numpy as numpy
from classifier import train_predict


def XGBstart(X_train, y_train, X_test, y_test):
    clf_B = xgb.XGBClassifier(seed=82)
    train_predict(clf_B, X_train, y_train, X_test, y_test)

