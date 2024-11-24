import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
import time
import missingno as msno
import seaborn as sns


# Function to make and visualize the graphs on screen
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
    #Load the datasets
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')

    #view_missing_data(train_data)

    X_train, X_test, y_train, y_test = divideDataset(train_data)

    #model = linearRegression(X_train, y_train, X_test, y_test)
    #model = val_poly_regression(X_train, y_train, X_test, y_test)
    #model = val_knn_regression(X_train, y_train, X_test, y_test)

    # Scale the features for gradient descent
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    test_data_scaled = scaler.transform(test_data.drop(columns=["id", "GeneticRisk", "ComorbidityIndex", "TreatmentResponse"]))

    m, b = gradient_descent_linear_regression(X_train_scaled, y_train, c=np.zeros_like(y_train), lr=0.01, epochs=1000)
    # m, b = gradient_descent_ridge(X_train_scaled, y_train, c=np.zeros_like(y_train), lr=0.01, epochs=1000, lambda_=0.1)
    # m, b = gradient_descent_lasso(X_train_scaled, y_train, c=np.zeros_like(y_train), lr=0.01, epochs=1000, lambda_=0.1)

    # Make predictions on the test set
    y_test_pred = X_test_scaled @ m + b

    # Calculate metrics using the custom error metric
    c = np.zeros_like(y_test)  # Since we are working with uncensored data
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Test Error Metric: {test_error:.4f}")

    # Plot y-y hat plot for test data
    plot_y_yhat(y_test, y_test_pred, plot_title="gradient_descent_model")
    create_csv(m, b, test_data_scaled)


def divideDataset(train_data):
    train_data_clean = train_data.dropna(subset=['SurvivalTime'])  # Remove rows with missing 'SurvivalTime'
    train_data_clean = train_data_clean[train_data_clean['Censored'] == 0]  # Keep only uncensored data points
    train_data_clean = train_data_clean.dropna(axis=1)  # Drop columns with any missing values
    print("Number of remaining points:", len(train_data_clean))

    X = train_data_clean.drop(columns=['SurvivalTime', 'Censored', 'id'])
    y = train_data_clean['SurvivalTime'].values

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(
        "Using a train-validation-test split allows for testing the model on both validation and test sets separately, providing a better performance estimate on unseen data.")
    print(
        "Cross-validation is more data-efficient for smaller datasets, as it uses the entire dataset in a rotating fashion for training and validation, rather than setting aside a fixed validation set.")

    return X_train, X_test, y_train, y_test


# function with the linear regression model of the problem
def linearRegression(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    # Calculate metrics using the custom error metric
    c = np.zeros_like(y_test)  # Since we are working with uncensored data
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    # Plotting y-y hat plot for test data
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

    # Calculate metrics using the custom error metric
    c = np.zeros_like(y_test)  # Since we are working with uncensored data
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    # Plotting y-y hat plot for test data
    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="polynomial_model")

    return pipeline


def val_knn_regression(X_train, y_train, X_test, y_test, k=30):
    pipeline = KNeighborsRegressor(n_neighbors=k)
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=10)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    # Calculate metrics using the custom error metric
    c = np.zeros_like(y_test)  # Since we are working with uncensored data
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    # Plotting y-y hat plot for test data
    plot_y_yhat(np.expand_dims(y_test, axis=-1), np.expand_dims(y_test_pred, axis=-1), plot_title="polynomial_model")

    return pipeline


def gradient_descent_linear_regression(X, y, c, lr=0.001, epochs=1000, clip_value=1.0):
    np.random.seed(42)
    w = np.random.randn(X.shape[1]) * 0.01  # Small random initialization
    b = 0.0

    n = len(y)
    for epoch in range(epochs):
        predictions = X @ w + b
        gradients_w = np.zeros_like(w)
        gradients_b = 0

        for i in range(n):
            error = y[i] - predictions[i]
            if c[i] == 0:  # Uncensored
                gradients_w += -2 * error * X[i] / n
                gradients_b += -2 * error / n
            else:  # Censored
                if error > 0:
                    gradients_w += -2 * error * X[i] / n
                    gradients_b += -2 * error / n

        # Clip gradients to avoid overflow
        gradients_w = np.clip(gradients_w, -clip_value, clip_value)
        gradients_b = np.clip(gradients_b, -clip_value, clip_value)

        # Update weights and bias
        w -= lr * gradients_w
        b -= lr * gradients_b

        # Compute the loss
        cMSE_loss = error_metric(y, predictions, c)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: cMSE Loss = {cMSE_loss:.4f}")

    return w, b


def gradient_descent_ridge(X, y, c, lr=0.01, epochs=1000, lambda_=0.1, clip_value=1.0):
    np.random.seed(42)
    m = np.random.randn(X.shape[1]) * 0.01
    b = np.random.randn()

    n = len(y)
    for epoch in range(epochs):
        predictions = X @ m + b
        gradients_m = np.zeros_like(m)
        gradients_b = 0

        for i in range(n):
            error = y[i] - predictions[i]
            if c[i] == 0:  # Uncensored
                gradients_m += -2 * error * X[i] / n
                gradients_b += -2 * error / n
            else:  # Censored
                if error > 0:
                    gradients_m += -2 * error * X[i] / n
                    gradients_b += -2 * error / n

        # Add L2 regularization gradient
        gradients_m += 2 * lambda_ * m
        gradients_m = np.clip(gradients_m, -clip_value, clip_value)
        gradients_b = np.clip(gradients_b, -clip_value, clip_value)

        # Update weights and bias
        m -= lr * gradients_m
        b -= lr * gradients_b

        if epoch % 100 == 0:
            cMSE_loss = error_metric(y, predictions, c) + lambda_ * np.sum(m ** 2)
            print(f"Epoch {epoch}: cMSE Loss = {cMSE_loss:.4f}")

    return m, b


def gradient_descent_lasso(X, y, c, lr=0.01, epochs=1000, lambda_=0.1, clip_value=1.0):
    np.random.seed(42)
    m = np.random.randn(X.shape[1]) * 0.01
    b = np.random.randn()

    n = len(y)
    for epoch in range(epochs):
        predictions = X @ m + b
        gradients_m = np.zeros_like(m)
        gradients_b = 0

        for i in range(n):
            error = y[i] - predictions[i]
            if c[i] == 0:  # Uncensored
                gradients_m += -2 * error * X[i] / n
                gradients_b += -2 * error / n
            else:  # Censored
                if error > 0:
                    gradients_m += -2 * error * X[i] / n
                    gradients_b += -2 * error / n

        # Add L1 regularization gradient
        gradients_m += lambda_ * np.sign(m)
        gradients_m = np.clip(gradients_m, -clip_value, clip_value)
        gradients_b = np.clip(gradients_b, -clip_value, clip_value)

        # Update weights and bias
        m -= lr * gradients_m
        b -= lr * gradients_b

        if epoch % 100 == 0:
            cMSE_loss = error_metric(y, predictions, c) + lambda_ * np.sum(np.abs(m))
            print(f"Epoch {epoch}: cMSE Loss = {cMSE_loss:.4f}")

    return m, b


def create_csv(w, b, test_data):
    result = test_data @ w + b
    result = pd.DataFrame(result, columns=['SurvivalTime'])
    result.index.name = "id"
    result.to_csv('cMSE-baseline-submission-02.csv')


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


#DONE!