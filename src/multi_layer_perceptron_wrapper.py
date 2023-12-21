import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor

from regressor_base import RegressorBase


class MultiLayerPerceptronWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(file_path, output_filename="./output_files/mlp_model.pkl")

    # def preprocess_data(self):
    #     self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    #     acronym_2d = self.data["acronym"].values.reshape(-1, 1)

    #     encoded_acronym = self.encoder.fit_transform(acronym_2d)

    #     encoded_acronym_df = pd.DataFrame(
    #         encoded_acronym,
    #         columns=self.encoder.get_feature_names_out(["acronym"]),
    #     )

    #     self.data = pd.concat([self.data, encoded_acronym_df], axis=1)

    #     self.data.drop("acronym", axis=1, inplace=True)

    #     self.X = self.data.drop("concentration", axis=1)
    #     self.y = self.data["concentration"]

    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         self.X, self.y, test_size=0.2, random_state=42
    #     )

    def train_model(self):
        self.model = MLPRegressor(
            activation="logistic",
            alpha=0.0001,
            hidden_layer_sizes=(100, 50),
            learning_rate="constant",
            learning_rate_init=0.01,
            max_iter=500,
            random_state=42,
        )

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred

    # def preprocess_data_to_predict(self, acronyms, codes):
    #     data = pd.DataFrame({"acronym": acronyms, "code": codes})

    #     acronym_2d = data["acronym"].values.reshape(-1, 1)

    #     encoded_acronym = self.encoder.transform(acronym_2d)
    #     encoded_acronym_df = pd.DataFrame(
    #         encoded_acronym,
    #         columns=self.encoder.get_feature_names_out(["acronym"]),
    #     )

    #     data = pd.concat([data, encoded_acronym_df], axis=1)

    #     data.drop("acronym", axis=1, inplace=True)
    #     return data
