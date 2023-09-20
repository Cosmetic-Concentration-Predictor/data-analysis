import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

from regressor_base import RegressorBase


class NeuralNetworkWrapper(RegressorBase):
    def train_model(self):
        model = keras.Sequential(
            [
                layers.Input(shape=(self.X_train_scaled.shape[1],)),
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(
            self.X_train_scaled, self.y_train, epochs=100, batch_size=32, verbose=0
        )

        self.model = model

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred

    def evaluate(self):
        try:
            mse = mean_squared_error(self.y_test, self.y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            r2 = r2_score(self.y_test, self.y_pred)
            self.mse = mse
            self.rmse = rmse
            self.mae = mae
            self.r2 = r2
        except ValueError:
            print(f"{self.y_test} - {self.y_pred}")

    def print_results(self):
        print("Neural Network Results:")
        print("MSE:", self.mse)
        print("RMSE:", self.rmse)
        print("MAE:", self.mae)
        print("R²:", self.r2)
