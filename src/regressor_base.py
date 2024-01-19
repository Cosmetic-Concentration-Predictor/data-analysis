import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder


class RegressorBase:
    def __init__(self, file_path, output_model=None, output_encoder=None):
        self.data = pd.read_csv(file_path)
        self.output_model = output_model
        self.output_encoder = output_encoder
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        self.encoder = None

    def preprocess_data(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        acronym_2d = self.data["acronym"].values.reshape(-1, 1)

        encoded_acronym = self.encoder.fit_transform(acronym_2d)

        encoded_acronym_df = pd.DataFrame(
            encoded_acronym,
            columns=self.encoder.get_feature_names_out(["acronym"]),
        )

        self.data = pd.concat([self.data, encoded_acronym_df], axis=1)

        self.data.drop("acronym", axis=1, inplace=True)

        self.X = self.data.drop("concentration", axis=1)
        self.y = self.data["concentration"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def preprocess_data_to_predict(self, acronyms, codes):
        data = pd.DataFrame({"acronym": acronyms, "code": codes})

        acronym_2d = data["acronym"].values.reshape(-1, 1)

        encoded_acronym = self.encoder.transform(acronym_2d)
        encoded_acronym_df = pd.DataFrame(
            encoded_acronym,
            columns=self.encoder.get_feature_names_out(["acronym"]),
        )

        data = pd.concat([data, encoded_acronym_df], axis=1)

        data.drop("acronym", axis=1, inplace=True)
        return data

    # def preprocess_data(self):
    #     self.X = self.data.drop("concentration", axis=1)
    #     self.y = self.data["concentration"]

    #     self.encoder = LabelEncoder()
    #     self.X["acronym"] = self.encoder.fit_transform(self.X["acronym"])

    #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    #         self.X, self.y, test_size=0.2, random_state=42
    #     )

    # def preprocess_data_to_predict(self, acronyms, codes):
    #     data = pd.DataFrame({"acronym": acronyms, "code": codes})
    #     data["acronym"] = self.encoder.transform(data["acronym"])
    #     return data

    def train_model(self):
        pass

    def predict(self):
        pass

    def save_configuration(self):
        if self.output_model is None:
            print("Provide a filename to save the model.")
            return

        if self.output_encoder is None:
            print("Provide a filename to save the encoder.")
            return

        if self.model:
            joblib.dump(self.model, self.output_model)
            print("Model saved successfully.")
        else:
            print("No model to save. Train the model first.")

        if self.encoder:
            joblib.dump(self.encoder, self.output_encoder)
            print("Encoder saved successfully.")
        else:
            print("No encoder to save.")

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
        print("Résultats:")
        print("MSE:", self.mse)
        print("RMSE:", self.rmse)
        print("MAE:", self.mae)
        print("R²:", self.r2)

    def plot_graphs(self):
        residuos = self.y_test - self.y_pred.reshape(-1)
        plt.scatter(self.y_pred, residuos)
        plt.xlabel("Prédictions")
        plt.ylabel("Résidus")
        plt.title("Graphique de dispersion des résidus par rapport aux prédictions")
        plt.show()

        sns.histplot(residuos, bins=30, kde=True)
        plt.xlabel("Résidus")
        plt.ylabel("Fréquence")
        plt.title("Histogrammes des résidus")
        plt.show()

        stats.probplot(residuos, dist="norm", plot=plt)
        plt.title("Graphiques QQ (quantile-quantile) des résidus")
        plt.show()

    def predict_multiple(self, acronyms, codes):
        input_data = self.preprocess_data_to_predict(acronyms, codes)
        self.predictions = self.model.predict(input_data)

        print(f"Predictions: {self.predictions}")

    def run(self):
        self.preprocess_data()
        self.train_model()
        self.predict()
        self.save_configuration()
        self.evaluate()
        self.print_results()
        self.plot_graphs()
