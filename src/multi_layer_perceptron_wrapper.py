import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers

from regressor_base import RegressorBase


class MultiLayerPerceptronWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(file_path, output_filename="./output_files/mlp_model.pkl")

    def preprocess_data(self):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        acronym_2d = self.data["acronym"].values.reshape(-1, 1)

        encoded_acronym = encoder.fit_transform(acronym_2d)

        encoded_acronym_df = pd.DataFrame(
            encoded_acronym,
            columns=encoder.get_feature_names_out(["acronym"]),
        )

        self.data = pd.concat([self.data, encoded_acronym_df], axis=1)

        self.data.drop("acronym", axis=1, inplace=True)

        self.X = self.data.drop("concentration", axis=1)
        self.y = self.data["concentration"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        self.model = keras.Sequential(
            [
                layers.Input(shape=(self.X_train.shape[1],)),
                layers.Dense(32, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ]
        )

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, verbose=0)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
