import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import missingno as msno


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

    X_train, X_test, y_train, y_test = divideDataset(train_data)

    model = linearRegression(X_train, y_train, X_test, y_test)

    test_data = test_data.drop(columns=["id", "GeneticRisk", "ComorbidityIndex", "TreatmentResponse"])
    create_csv(model, test_data)


def divideDataset(train_data):
    train_data_clean = train_data.dropna(subset=['SurvivalTime'])
    train_data_clean = train_data_clean[train_data_clean['Censored'] == 0]
    train_data_clean = train_data_clean.dropna(axis=1)
    print("Number of remaining points:", len(train_data_clean))

    #sns.pairplot(train_data_clean, y_vars=['SurvivalTime'], x_vars=train_data_clean.columns.drop(['SurvivalTime', 'Censored']))
    #plt.show()

    X = train_data_clean.drop(columns=['SurvivalTime', 'Censored', 'id'])
    y = train_data_clean['SurvivalTime'].values

    #X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
    #X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Using a train-validation-test split allows for testing the model on both validation and test sets separately, providing a better performance estimate on unseen data.")
    print("Cross-validation is more data-efficient for smaller datasets, as it uses the entire dataset in a rotating fashion for training and validation, rather than setting aside a fixed validation set.")

    return X_train, X_test, y_train, y_test


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


def create_csv(model, test_data):
    result = model.predict(test_data)
    result = pd.DataFrame(result).rename(columns={0:'0'})
    result.index.name = "id"
    result.to_csv('baseline-submission-01.csv')


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
