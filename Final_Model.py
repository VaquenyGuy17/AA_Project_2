import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
import seaborn as sns
import missingno as msno


#function to make and visualize the graphs on screen
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


# Main function to train and evaluate models
def baselineModel():
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    X_train, X_test, y_train, y_test, X_labeled, X_unlabeled = divideDataset(train_data)

    print("Training Semi-supervised model...")
    model = semi_supervised(X_train, y_train, X_test, y_test, X_labeled, X_unlabeled)

    test_data = test_data.drop(columns=["id"])
    create_csv(model, test_data)


# Function to divide and process the dataset
def divideDataset(train_data):
    labeled_data = train_data.dropna(subset=['SurvivalTime'])
    unlabeled_data = train_data[train_data['SurvivalTime'].isnull()]

    print(f"Labeled data: {labeled_data.shape}")
    print(f"Unlabeled data: {unlabeled_data.shape}")

    combined_data = pd.concat([labeled_data, unlabeled_data], ignore_index=True)
    combined_data = handle_missing_data(combined_data)

    labeled_data = combined_data.loc[combined_data['SurvivalTime'].notnull()]
    unlabeled_data = combined_data.loc[combined_data['SurvivalTime'].isnull()]

    X_labeled = labeled_data.drop(columns=['SurvivalTime', 'Censored', 'id'])
    y_labeled = labeled_data['SurvivalTime'].values

    X_unlabeled = unlabeled_data.drop(columns=['SurvivalTime', 'Censored', 'id'])

    X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X_labeled, X_unlabeled


# Function to handle missing data
def handle_missing_data(data):
    imputer = IterativeImputer(max_iter=10)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    return data


# Function for Semi-supervised learning
def semi_supervised(X_train, y_train, X_test, y_test, X_labeled, X_unlabeled):
    pipeline = HistGradientBoostingRegressor(max_iter=15)
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    # Calculate metrics using the custom error metric
    c = np.zeros_like(y_test)  # Since we are working with uncensored data
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(y_test, y_test_pred, plot_title="semi_supervised_model")
    return pipeline


def error_metric(y, y_hat, c):
    import numpy as np
    err = y-y_hat
    err = (1-c)*err**2 + c*np.maximum(0,err)**2
    return np.sum(err)/err.shape[0]


def create_csv(model, test_data):
    result = model.predict(test_data)
    result = pd.DataFrame(result).rename(columns={0:'0'})
    result.index.name = "id"
    result.to_csv('optional-submission-01.csv.csv')


baselineModel()
