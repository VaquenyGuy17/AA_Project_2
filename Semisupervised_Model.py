import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import missingno as msno
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor


def plot_y_yhat(y_val, y_pred, plot_title="plot"):
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val), MAX, replace=False)
    else:
        idx = np.arange(len(y_val))

    plt.figure(figsize=(8, 8))
    x0 = np.min(y_val[idx])
    x1 = np.max(y_val[idx])

    plt.scatter(y_val[idx], y_pred[idx])
    plt.xlabel('True Survival Time')
    plt.ylabel('Predicted Survival Time')
    plt.plot([x0, x1], [x0, x1], color='black', linestyle="-", linewidth=3)
    plt.axis('square')
    plt.title(plot_title)
    plt.savefig(plot_title + '.pdf')
    plt.show()


def baselineModel():
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    #view_missing_data(train_data)

    X_train, X_test, y_train, y_test, X_labeled, X_unlabeled = divideDataset(train_data)

    #print("Training HistGradientBoostingRegressor...")
    #model = train_hist_gradient_boosting(X_train, y_train, X_test, y_test)

    #print("Training CatBoostRegressor...")
    #model = train_catboost(X_train, y_train, X_test, y_test)

    #model = linearRegression(X_train, y_train, X_test, y_test)
    #model = val_poly_regression(X_train, y_train, X_test, y_test)
    #model = val_knn_regression(X_train, y_train, X_test, y_test)

    model = semi_supervised(X_train, y_train, X_test, y_test, X_labeled, X_unlabeled)

    test_data = test_data.drop(columns=["id"])
    create_csv(model, test_data)


def divideDataset(train_data):
    labeled_data = train_data.dropna(subset=['SurvivalTime'])
    unlabeled_data = train_data[train_data['SurvivalTime'].isnull()]

    print("Labeled data size:", labeled_data.shape)
    print("Unlabeled data size:", unlabeled_data.shape)

    combined_data = pd.concat([labeled_data, unlabeled_data], ignore_index=True)

    combined_data = handle_missing_data(combined_data, "iterative")

    labeled_data = combined_data.loc[combined_data['SurvivalTime'].notnull()]
    unlabeled_data = combined_data.loc[combined_data['SurvivalTime'].isnull()]

    X_labeled = labeled_data.drop(columns=['SurvivalTime', 'Censored', 'id'])
    y_labeled = labeled_data['SurvivalTime'].values

    X_unlabeled = unlabeled_data.drop(columns=['SurvivalTime', 'Censored', 'id'])

    X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2)
    return X_train, X_test, y_train, y_test, X_labeled, X_unlabeled


def linearRegression(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="linear_regression_baseline")

    return pipeline


def val_poly_regression(X_train, y_train, X_test, y_test, regressor=RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1]), degree=2, max_features=None):
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('scaler', StandardScaler()),
        ('model', regressor)
    ])
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="polynomial_model")

    return pipeline


def val_knn_regression(X_train, y_train, X_test, y_test, k=30):
    pipeline = KNeighborsRegressor(n_neighbors=k)
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="polynomial_model")

    return pipeline


def handle_missing_data(train_data, method):

    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'iterative':
        imputer = IterativeImputer(max_iter=10)
    else:
        raise ValueError("Invalid imputation.")

    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    train_data[numeric_cols] = imputer.fit_transform(train_data[numeric_cols])
    return train_data


def train_hist_gradient_boosting(X_train, y_train, X_test, y_test):
    pipeline = HistGradientBoostingRegressor(max_iter=100)
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="polynomial_model")

    return pipeline


def train_catboost(X_train, y_train, X_test, y_test):
    pipeline = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, silent=True)
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="polynomial_model")

    return pipeline


class FrozenTransformer(BaseEstimator):
    def __init__(self, fitted_transformer):
        self.fitted_transformer = fitted_transformer

    def __getattr__(self, name):
        return getattr(self.fitted_transformer, name)

    def __sklearn_clone__(self):
        return self

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self.fitted_transformer.transform(X)

    def fit_transform(self, X, y=None):
        return self.fitted_transformer.transform(X)


def semi_supervised(X_train, y_train, X_test, y_test, X_labeled, X_unlabeled):
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(pd.concat([X_labeled, X_unlabeled]))
    isomap = Isomap(n_components=9)
    X_combined_reduced = isomap.fit_transform(X_combined_scaled)
    frozen_isomap = FrozenTransformer(isomap)

    pipeline = Pipeline([
        ('imputer', IterativeImputer(max_iter=10)),
        ('scaler', StandardScaler()),
        ('isomap', frozen_isomap),
        ('regressor', HistGradientBoostingRegressor(max_iter=100))
    ])
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(y_test, y_test_pred, plot_title="semi_supervised_model")
    return pipeline


def create_csv(model, test_data):
    result = model.predict(test_data)
    result = pd.DataFrame(result).rename(columns={0:'0'})
    result.index.name = "id"
    result.to_csv('semisupervised-submission-10.csv')


def view_missing_data(train_data):
    msno.bar(train_data)
    msno.heatmap(train_data)
    msno.matrix(train_data)
    msno.dendrogram(train_data)
    plt.show()


def error_metric(y, y_hat, c):
    import numpy as np
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


baselineModel()
