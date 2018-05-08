import xgboost as xgb
from sklearn.metrics import f1_score



def run_xgboost(X, y, X_test, y_test):
    clf = xgb.XGBClassifier()
    
    clf.fit(X, y)

    y_pred_train = clf.predict(X)
    y_pred_test = clf.predict(X_test)

    f1_train = f1_score(y, y_pred_train, pos_label="failed")
    acc_train = sum(y == y_pred_train) / float(len(y_pred_train))


    print("XGBoost On the training set:")
    print(f1_train)
    print(acc_train)

    f1_test = f1_score(y_test, y_pred_test, pos_label="failed")
    acc_test = sum(y_test == y_pred_test) / float(len(y_pred_test))

    print("XGBoost On the testing set:")
    print(f1_test)
    print(acc_test)

    return clf