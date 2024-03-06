import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

    # def preprocess_data(self):
    #     self.X = self.data.drop("concentration", axis=1)
    #     self.y = self.data["concentration"]

    #     self.encoder = LabelEncoder()
    #     self.X["acronym"] = self.encoder.fit_transform(self.X["acronym"])

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
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, rmse, mae, r2

    def k_fold_cross_validation(self, num_folds=5):
        print("Starting k fold cross validation")
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        metrics = {"mse": [], "rmse": [], "mae": [], "r2": []}

        for train_index, test_index in kf.split(self.X):
            self.X_train, self.X_test = (
                self.X.iloc[train_index],
                self.X.iloc[test_index],
            )
            self.y_train, self.y_test = (
                self.y.iloc[train_index],
                self.y.iloc[test_index],
            )
            self.train_model()
            mse, rmse, mae, r2 = self.evaluate()
            metrics["mse"].append(mse)
            metrics["rmse"].append(rmse)
            metrics["mae"].append(mae)
            metrics["r2"].append(r2)
        return metrics

    def calculate_metrics(self, metrics):
        self.mse_mean = np.mean(metrics["mse"])
        self.mse_std = np.std(metrics["mse"])
        self.rmse_mean = np.mean(metrics["rmse"])
        self.rmse_std = np.std(metrics["rmse"])
        self.mae_mean = np.mean(metrics["mae"])
        self.mae_std = np.std(metrics["mae"])
        self.r2_mean = np.mean(metrics["r2"])
        self.r2_std = np.std(metrics["r2"])

    def print_metrics(self):
        print(f"MSE Mean: {self.mse_mean}, MSE Std: {self.mse_std}")
        print(f"RMSE Mean: {self.rmse_mean}, RMSE Std: {self.rmse_std}")
        print(f"MAE Mean: {self.mae_mean}, MAE Std: {self.mae_std}")
        print(f"R2 Mean: {self.r2_mean}, R2 Std: {self.r2_std}")

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
        # metrics = self.k_fold_cross_validation()
        # self.calculate_metrics(metrics)
        self.train_model()
        self.predict()
        # self.print_metrics()
        # self.evaluate()
        # self.print_results()
        self.plot_graphs()
