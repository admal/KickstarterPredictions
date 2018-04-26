import operator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tpot import TPOTClassifier

from data import *
from logistic_regression import start


def main():
    data = load_data()
    total_data = len(data)
    total_successes = len(data[data.state == 'successful'])
    total_fails = len(data[data.state == 'failed'])
    total_other = total_data - total_successes - total_fails

    s_rate = round(float(total_successes) / total_data * 100, 2)
    f_rate = round(float(total_fails) / total_data * 100, 2)
    o_rate = round(float(total_other) / total_data * 100, 2)
    print("Total number of projects: {}".format(total_data))
    print("Successes: {} ({}%), fails: {} ({}%), other: {} ({}%)".format(total_successes, s_rate, total_fails, f_rate,
                                                                         total_other, o_rate))
    # # input data
    # x = data.drop(['state'], 1)
    # # output (state of project)
    # y = data['state']
    #
    # # y = preprocess_features(y)
    # # convert categorical data into classes
    # x = preprocess_features(x)
    # # print("Processed feature columns ({} total features):\n{}".format(len(x.columns), list(x.columns)))
    # features = list(x.columns)
    # print("Generated features: {}".format(features))
    #
    # categories = [label for label in features if
    #               operator.contains(label, "category") and not operator.contains(label, "main_category")]
    # print("Category count: {}".format(len(categories)))
    # print(categories)
    #
    # main_categories = [label for label in features if operator.contains(label, "main_category")]
    # print("Main category count: {}".format(len(main_categories)))
    # print(main_categories)
    #
    # currency = [label for label in features if operator.contains(label, "currency")]
    # print("Currency count: {}".format(len(currency)))
    # print(currency)
    #
    # country = [label for label in features if operator.contains(label, "country")]
    # print("Country count: {}".format(len(country)))
    # print(country)

    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

    data['state'] = data['state'].map({'successful': 0, 'failed': 1})
    data_class = data['state'].values

    features = data.drop('state', axis=1).values

    training_features, testing_features, training_target, testing_target = train_test_split(features,
                                                                                            data_class,
                                                                                            random_state=42,
                                                                                            train_size=0.8)

    exported_pipeline = LinearSVC(C=20.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.01)
    # exported_pipeline = LogisticRegression(random_state=42)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(training_features)
    print(np.sum(results == 0))
    print(np.sum(results == 1))
    print(np.sum(training_target == 0))
    print(np.sum(training_target == 1))
    acc = np.sum(results == training_target) / float(len(results))

    print("accuracy score for test set: {}.".format(acc))
    print(results)


#
# tpot = TPOTClassifier(generations=1, verbosity=2)
# tpot.fit(data.drop('state', axis=1).loc[training_indicies].values, data.loc[training_indicies, 'state'].values)
# tpot.score(data.drop('state', axis=1).loc[validation_indicies].values,
#            data.loc[validation_indicies, 'state'].values)
# tpot.export('pipeline.py')

# Shuffle and split the dataset into training and testing set.
# X_train, X_test, y_train, y_test = train_test_split(x, y,
#                                                     test_size=0.1,
#                                                     random_state=2,
#                                                     stratify=y)

# start(X_train.values, y_train.tolist(), X_test.values, y_test.tolist())

print('elo')

if __name__ == '__main__':
    main()
