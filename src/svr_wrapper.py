import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

from regressor_base import RegressorBase


class SVRWrapper(RegressorBase):
    def train_model(self):
        self.model = SVR(kernel="linear", C=1.0, epsilon=0.1)

        self.model.fit(self.X_train_scaled, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred

    def evaluate(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        self.mse = mse
        self.rmse = rmse
        self.mae = mae
        self.r2 = r2

    def print_results(self):
        print("SVR Results:")
        print("MSE:", self.mse)
        print("RMSE:", self.rmse)
        print("MAE:", self.mae)
        print("R²:", self.r2)
