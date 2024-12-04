import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor, \
    HistGradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore")


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


def handle_missing_data(train_data, method="iterative"):
    if method == 'mean':
        train_data.fillna(train_data.mean(), inplace=True)
    elif method == 'median':
        train_data.fillna(train_data.median(), inplace=True)
    elif method == 'iterative':
        imputer = IterativeImputer(max_iter=10)
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        train_data[numeric_cols] = imputer.fit_transform(train_data[numeric_cols])
    else:
        raise ValueError("Invalid imputation method specified.")
    return train_data


def divideDataset(train_data):
    train_data = handle_missing_data(train_data, "iterative")
    train_data_clean = train_data.dropna(subset=['SurvivalTime'])
    train_data_clean = train_data_clean[train_data_clean['Censored'] == 0]
    print("Number of remaining points:", len(train_data_clean))

    X = train_data_clean.drop(columns=['SurvivalTime', 'Censored', 'id'])
    y = train_data_clean['SurvivalTime'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



def train_model(X_train, y_train, X_test, y_test):
    pipeline = HistGradientBoostingRegressor(max_iter=100)
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")
    plot_y_yhat(y_test, y_test_pred, "Stacking Regressor")
    return pipeline


def error_metric(y, y_hat, c):
    import numpy as np
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


def baselineModel():
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    X_train, X_test, y_train, y_test = divideDataset(train_data)
    model = train_model(X_train, y_train, X_test, y_test)

    test_data_clean = handle_missing_data(test_data, "iterative").drop(columns=["id"])
    result = model.predict(test_data_clean)
    result = pd.DataFrame(result).rename(columns={0: '0'})
    result.index.name = "id"
    result.to_csv('optional-submission-05.csv')



baselineModel()
