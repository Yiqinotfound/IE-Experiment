from sklearn.model_selection import train_test_split
import numpy as np


class LinearModel:
    def __init__(
        self,
        ridge: float = 0.0,
    ):
        self.ridge = ridge

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.N_train, self.D = X_train.shape
        X_b = np.hstack([np.ones((self.N_train, 1)), self.X_train])
        if self.ridge > 0:
            # Apply Ridge regression
            identity = np.eye(X_b.shape[1])
            theta = (
                np.linalg.inv(X_b.T @ X_b + self.ridge * identity)
                @ X_b.T
                @ self.y_train
            )
        else:
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ self.y_train
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

    def predict(self, X: np.ndarray):
        N = X.shape[0]
        X_b = np.hstack([np.ones((N, 1)), X])
        theta = np.hstack([self.intercept_, self.coef_])
        y_pred = X_b @ theta
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        # return rmse, mae, r2
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # mae
        mae = np.mean(np.abs(y - y_pred))

        # r2
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return rmse, mae, r2
