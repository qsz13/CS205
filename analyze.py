from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

plot_path = "plot/"
file_path = 'data.csv'


def load_raw_data():
    df = pd.read_csv(file_path)
    df.drop('id', axis=1, inplace=True)

    return df


def preprocess(data):
    data = data.sample(frac=1)
    data['gender'] -= 1
    data['injury'] -= 1
    data, target = data.drop(['jump', 'color'], axis=1), data['jump']
    data = pd.get_dummies(data, columns=["race"])
    min_max = MinMaxScaler()
    scaled = min_max.fit_transform(data)
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)
    return data, target


def plot_numeric(data, x, y, hue='gender'):
    sns.lmplot(x=x, y=y, hue=hue, data=data, legend=False)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True)
    lgd.get_frame().set_color("white")
    plt.savefig(plot_path + '%s - %s.png' % (x, y), dpi=300, additional_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def plot_categorical(data, x, y, hue='gender'):
    sns.swarmplot(x=x, y=y, hue=hue, data=data)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    lgd.get_frame().set_color("white")
    plt.savefig(plot_path + '%s - %s.png' % (x, y), dpi=300, additional_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def readable(df):
    df = df.copy()
    df['gender'].replace([1, 2], ['Female', 'Male'], inplace=True)
    df['race'].replace([1, 4, 5, 6, 8, 9], ['White', 'Indian', 'Chinese', 'Korean', 'Two or more', 'Other'],
                       inplace=True)
    df['injury'].replace([1, 2], ['injured', 'not injured'], inplace=True)
    df['color'].replace([1, 2, 3, 4, 5, 6, 7], ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow,', 'Other'],
                        inplace=True)
    df['competitive'].replace([1, 2, 3], ['Agree', 'Neither', 'Disagree'], inplace=True)
    return df


def plot_all(data):
    data = data.copy()
    min_max = MinMaxScaler()
    data[['height', 'exercise', 'competitive', 'weight', 'age', 'sleep', 'color', 'race']] = \
        min_max.fit_transform(data[['height', 'exercise', 'competitive', 'weight', 'age', 'sleep', 'color', 'race']])
    data = data.sort_values(by=['jump'], ascending=[True])

    plt.subplot(3, 1, 1)
    data[['height', 'weight', 'age']].plot(use_index=False, ax=plt.gca())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True)

    plt.subplot(3, 1, 2)
    data[['exercise', 'competitive', 'sleep', 'color', 'race']].plot(use_index=False, ax=plt.gca())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True)

    plt.subplot(3, 1, 3)
    data['jump'].plot(use_index=False, ax=plt.gca())
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=True)

    plt.savefig(plot_path + 'plot.png', dpi=300, bbox_inches='tight')
    plt.clf()


def plot(data):
    readable_data = readable(data)
    for x in ['age', 'exercise', 'height', 'weight', 'sleep']:
        plot_numeric(readable_data, x, 'jump')
    for x in ['competitive', 'gender', 'injury', 'color', 'race']:
        plot_categorical(readable_data, x, 'jump')
    plot_all(data)


class MeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.ret = 0

    def fit(self, X, y, **kwargs):
        self.ret = np.median(y)

    def predict(self, X):
        return [self.ret for x in X]


def linear_analysis(data, target):
    model = sm.OLS(target, data)
    results = model.fit()
    print(results.summary())
    print results.pvalues.sort_values()


def evaluate(model, data, target, method, param_grid=None, n_jobs=1, random_state=1):
    print method
    if param_grid == None:
        # score = cross_val_score(model, data, target, cv=13, scoring='r2')
        # print "R^2: " + str(score)
        score = cross_val_score(model, data, target, cv=KFold(4,shuffle=True), scoring='neg_mean_absolute_error')
        # print "MAE: " + str(score)
        print "Mean MAE: " + str(np.mean(score))
        print "stdev:"+str(np.std(score))
        # score = cross_val_score(model, data, target, cv=13, scoring='neg_mean_squared_error')
        # print "RMSE:" + str(np.sqrt(-np.mean(score)))
        return np.mean(score), np.std(score)
    else:
        clf = GridSearchCV(model, param_grid=param_grid, scoring="neg_mean_absolute_error", cv=13, n_jobs=n_jobs)
        clf.fit(data, target)
        print "Best MAE: ", clf.best_score_
        print clf.best_params_
        model.set_params(**clf.best_params_)
        score = cross_val_score(model, data, target, cv=13)
        print "R^2:" + str(score)
    print ""


def lassoCV(data, target):
    model = LassoCV(cv=4, max_iter=100000, normalize=True).fit(data, target)

    # Display results
    m_log_alphas = -np.log10(model.alphas_)
    plt.figure()
    ymin, ymax = 2000, 5000
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent ')
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    plt.savefig(plot_path + 'lassoCV.png', dpi=300)

    return model.alpha_


def lasso_lars_cv(data, target):
    print("Computing regularization path using the Lars lasso...")
    model = LassoLarsCV(cv=4).fit(data, target)

    # Display results
    m_log_alphas = -np.log10(model.cv_alphas_)
    plt.figure()
    ymin, ymax = 0, 5000
    plt.plot(m_log_alphas, model.cv_mse_path_, ':')
    plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha CV')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: Lars')
    plt.axis('tight')
    plt.ylim(ymin, ymax)

    plt.savefig(plot_path + 'lasso_lars_cv.png', dpi=300)
    plt.clf()
    return model.alpha_


def lasso_lars_ic(data, target):
    model_bic = LassoLarsIC(criterion='bic')
    model_bic.fit(data, target)
    alpha_bic_ = model_bic.alpha_

    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(data, target)
    alpha_aic_ = model_aic.alpha_

    def plot_ic_criterion(model, name, color):
        alpha_ = model.alpha_
        alphas_ = model.alphas_
        criterion_ = model.criterion_
        plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
        plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel('-log(alpha)')
        plt.ylabel('criterion')

    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plot_ic_criterion(model_bic, 'BIC', 'r')
    plt.legend()
    plt.title('Information-criterion for model selection')
    plt.savefig(plot_path + 'lasso_lars.png', dpi=300)


def model_compare(data, target):
    # # mean
    # evaluate(MeanEstimator(), data.values, target.values, "Mean Estimator")
    #
    # # linear
    # param_grid = {"normalize": [True, False], "fit_intercept": [True, False]}
    # evaluate(LinearRegression(), data, target, "Linear", param_grid=param_grid)
    #
    # # poly
    # poly = Pipeline([('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    #                  ('linear', Lasso(alpha=3))])
    # evaluate(poly, data, target, "Poly")
    #
    # # decision tree
    # param_grid = {"max_features": ["auto", "sqrt", "log2", None]}
    # evaluate(DecisionTreeRegressor(criterion="mae"), data, target, "Decision Tree", param_grid=param_grid)
    #
    # # elastic
    # param_grid = dict(alpha=10.0 ** np.arange(-5, 4), l1_ratio=0.1 * np.arange(0, 11), normalize=[True, False],
    #                   fit_intercept=[True, False])
    # l = []
    # for i in range(1000):
    #     print i
    #     en = ElasticNet(normalize=False, alpha= 0.1, l1_ratio=0.2, fit_intercept= True)
    #
    #     l.append(evaluate(en, data, target, "Elastic"))
    # print np.mean(l)
    #
    # # ridge
    # param_grid = dict(alpha=10.0 ** np.arange(-5, 4), normalize=[True, False], fit_intercept=[True, False],
    #                   solver=["auto", "svd", "cholesky", "lsqr", 'sparse_cg', 'sag'])
    # evaluate(Ridge(), data, target, "Ridge", param_grid=param_grid)

    ri = Ridge(normalize=False, alpha=1.0, solver='sparse_cg', fit_intercept=True)
    l  = []
    for i in range(1000):
        print i

        l.append(evaluate(ri, data, target, "Elastic"))
    print np.mean(l)

    #
    # SVR
    # param_grid = dict(C=10.0 * np.arange(50, 70, 10), kernel=['linear', 'poly', 'rbf', 'sigmoid'])
    # evaluate(svm.SVR(), data, target, "SVR", param_grid=param_grid, n_jobs=4)
    # svrmae = []
    # svrstd = []
    # lassomae = []
    # lassostd = []
    #
    # alpha = lassoCV(data, target)
    # # print "alpha " + str(alpha)
    #
    # lasso = Lasso(alpha=alpha, normalize=False)
    #
    # for i in range(1000):
    #     print i
    #     err , std= evaluate(svm.SVR(kernel= 'sigmoid', C= 500.0), data, target, "SVR", random_state=i)
    #     svrmae.append(err)
    #     svrstd.append(std)
    #     err, std = evaluate(lasso, data, target, "SVR", random_state=i)
    #     lassomae.append(err)
    #     lassostd.append(std)
    #
    #
    #
    # print np.mean(svrmae)
    # print np.mean(svrstd)
    # print np.mean(lassomae)
    # print np.mean(lassostd)
    #
    # # XGBoost
    # param_grid = {'max_depth': [2, 4, 6],
    #               'n_estimators': [50, 100, 200]}
    # evaluate(xgb.XGBRegressor(), data, target, "XGBoost", param_grid=param_grid)
    #
    # # SGD Regressor()
    # param_grid = {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    #               'penalty': ['none', 'l2', 'l1', 'elasticnet']}
    # evaluate(SGDRegressor(), data, target, "SGDRegressor", param_grid=param_grid)
    #
    # # GradientBoostingRegressor
    # gbr = GradientBoostingRegressor(loss='quantile', criterion="mae")
    # evaluate(gbr, data, target, "GradientBoostingRegressor")
    #
    # # # AdaBoostClassifier
    # # ada = AdaBoostClassifier(n_estimators=100)
    # # evaluate(ada, data, target, "AdaBoostClassifier")
    #
    # # BaggingRegressor
    # evaluate(BaggingRegressor(), data, target, "BaggingRegressor")
    #
    # # KNeighborsRegressor
    # kn = KNeighborsRegressor(n_neighbors=4, weights="distance")
    # evaluate(kn, data, target, "KNeighborsRegressor")
    #
    # # BayesianRidge
    # br = BayesianRidge()
    # evaluate(br, data, target, "BayesianRidge")

    # lasso_lars_ic(data, target)

    # alpha = lassoCV(data, target)
    # # print "alpha " + str(alpha)
    # lasso = Lasso(alpha=alpha, normalize=False)
    # evaluate(lasso, data, target, "Lasso")

    # alpha = lasso_lars_cv(data, target)
    # print "alpha " + str(alpha)
    # lasso_lars = LassoLars(alpha=alpha)
    # evaluate(lasso_lars, data, target, "Lasso Lars")


def main():
    df = load_raw_data()
    # plot(df)
    data, target = preprocess(df)
    # linear_analysis(data, target)

    # for i in range(len(data)):
    #     print "dropping ", i
    #     data1 , target1= data.drop(i),target.drop(i)
    model_compare(data, target)


if __name__ == '__main__':
    main()
