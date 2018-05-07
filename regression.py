from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def run_regression(x, y, x_test, y_test):
	clf = LogisticRegression(random_state=42)

	clf.fit(x, y)

	y_pred = clf.predict(x_test)
	y_pred_train = clf.predict(x)
	pos_label = 'failed'
	f1 = f1_score(y, y_pred_train, pos_label=pos_label)
	acc = sum(y == y_pred_train) / float(len(y_pred_train))

	print('Train')
	print(f1)
	print(acc)

	return f1_score(y_test, y_pred, pos_label=pos_label), sum(y_test == y_pred) / float(len(y_pred))