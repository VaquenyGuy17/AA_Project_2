import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
import missingno as msno
import seaborn as sns


# Load the datasets
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


# Visualize missing data
msno.bar(train_data)
msno.heatmap(train_data)
#msno.matrix(train_data)
#msno.dendrogram(train_data)
#plt.show()


# Custom error metric (cMSE)
def error_metric(y, y_hat, c):
    err = y - y_hat
    err = (1 - c) * err**2 + c * np.maximum(0, err)**2
    return np.sum(err) / err.shape[0]


# Divide dataset into train-test splits
def divideDataset(train_data):
    train_data_clean = train_data.dropna(subset=['SurvivalTime'])  # Remove rows with missing 'SurvivalTime'
    train_data_clean = train_data_clean[train_data_clean['Censored'] == 0]  # Keep only uncensored data points
    train_data_clean = train_data_clean.dropna(axis=1)  # Drop columns with any missing values
    print("Number of remaining points:", len(train_data_clean))

    X = train_data_clean.drop(columns=['SurvivalTime', 'Censored'])
    y = train_data_clean['SurvivalTime'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = divideDataset(train_data)


# Function to visualize predictions
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


# Gradient descent implementation for cMSE
def gradient_descent(X, y, c, lr=0.01, epochs=1000):
    np.random.seed(42)
    w = np.random.randn(X.shape[1])
    b = np.random.randn()

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

        # Update weights and bias
        w -= lr * gradients_w
        b -= lr * gradients_b

        if epoch % 100 == 0:
            cMSE_loss = error_metric(y, predictions, c)
            print(f"Epoch {epoch}: cMSE Loss = {cMSE_loss:.4f}")

    return w, b


# Linear regression model
def linearRegression(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    y_train_pred = cross_val_predict(pipeline, X_train, y_train, cv=5)
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    c = np.zeros_like(y_test)  # Uncensored data
    train_error = error_metric(y_train, y_train_pred, c=np.zeros_like(y_train))
    test_error = error_metric(y_test, y_test_pred, c)

    print(f"Train Error Metric: {train_error:.4f}")
    print(f"Test Error Metric: {test_error:.4f}")

    plot_y_yhat(y_test, y_test_pred, plot_title="linear_regression_baseline")


# Experiment with Ridge Regularization
def ridgeRegression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    ridge.fit(X_train_scaled, y_train)
    y_test_pred = ridge.predict(X_test_scaled)

    plot_y_yhat(y_test, y_test_pred, plot_title="ridge_regularization")

# Experiment with Lasso Regularization
def lassoRegression(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = LassoCV(alphas=[0.01, 0.1, 1.0], cv=5)
    lasso.fit(X_train_scaled, y_train)
    y_test_pred = lasso.predict(X_test_scaled)

    plot_y_yhat(y_test, y_test_pred, plot_title="lasso_regularization")

# Run all models
linearRegression(X_train, y_train, X_test, y_test)


# Prepare data for gradient descent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

c_train = train_data.loc[X_train.index, 'Censored'].values
weights, bias = gradient_descent(X_train_scaled, y_train, c=c_train, lr=0.01, epochs=1000)
y_test_pred_gd = X_test_scaled @ weights + bias
plot_y_yhat(y_test, y_test_pred_gd, plot_title="gradient_descent")

ridgeRegression(X_train, y_train, X_test, y_test)
lassoRegression(X_train, y_train, X_test, y_test)
