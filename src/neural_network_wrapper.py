from tensorflow import keras
from tensorflow.keras import layers

from regressor_base import RegressorBase


class NeuralNetworkWrapper(RegressorBase):
    def train_model(self):
        model = keras.Sequential(
            [
                layers.Input(shape=(self.X_train.shape[1],)),
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, verbose=0)

        self.model = model

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
