from sklearn import svm

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR

file_training_path = 'data.csv'
file_testing_path = 'test.csv'


def preprocess(data):
    data = data.drop(['color'], axis=1)
    data['race'] = data['race'].astype('category', categories=[1, 4, 5, 6, 8, 9])

    data = pd.get_dummies(data, columns=["race"])
    min_max = MinMaxScaler()
    scaled = min_max.fit_transform(data)
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)
    return data


def load_training_data():
    df = pd.read_csv(file_training_path)
    df.drop('id', axis=1, inplace=True)
    return preprocess(df.drop(['jump'], axis=1)), df[['jump']]


def load_testing_data():
    df = pd.read_csv(file_testing_path)
    df.drop('id', axis=1, inplace=True)
    return preprocess(df)


class MeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.ret = 0

    def fit(self, X, y, **kwargs):
        self.ret = np.mean(y)

    def predict(self, X):
        return np.array([self.ret for x in X])



def main():
    training_X, training_y = load_training_data()
    testing_X = load_testing_data()

    training_X.to_csv("./preprocess_train.csv")
    testing_X.to_csv("./preprocess_test.csv")

    # cv = LassoCV(cv=13, max_iter=100000, normalize=True).fit(training_X, training_y.values.ravel())
    # print cv.alpha_
    # lasso = Lasso(alpha=cv.alpha_, normalize=False)
    # lasso.fit(training_X, training_y.values.ravel())




    # lasso.fit(training_X.values, training_y.values)
    # # print training_X.columns
    # # for i in range(len(lasso.sparse_coef_.indices)):
    # #     print training_X.columns[lasso.sparse_coef_.indices[i]],lasso.sparse_coef_.data[i]
    # # print lasso.intercept_
    # prediction = lasso.predict(testing_X.values)
    # print prediction




    # en = ElasticNet(normalize=False, alpha= 0.1, l1_ratio=0.2, fit_intercept= True)
    # en.fit(training_X.values, training_y.values)
    # prediction = en.predict(testing_X.values)
    # print en.coef_

    ri = Ridge(normalize=False, alpha=1.0, solver='sparse_cg', fit_intercept= True)
    ri.fit(training_X.values, training_y.values)
    pre = ri.predict(testing_X.values)
    prediction = []
    for p in pre:
        prediction.append(p[0])
    prediction = np.array(prediction)


    # s = svm.SVR(kernel='sigmoid', C=500.0)
    # s.fit(training_X.values, training_y.values)
    # prediction = s.predict(testing_X.values)


    actual = [434,398,356,360,475,392,366,386,511,431,516]
    print "Error:", np.sum(np.abs(prediction-actual))

    print prediction


if __name__ == '__main__':
    main()


