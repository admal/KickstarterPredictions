from sklearn.linear_model import LogisticRegression

from classifiers import train_predict


def start(X_train, y_train, X_test, y_test):
    clf_A = LogisticRegression(random_state=42)
    train_predict(clf_A, X_train, y_train, X_test, y_test)
