import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class RegressorBase:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def preprocess_data(self):
        encoder = OneHotEncoder(sparse_output=False)

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
        pass

    def predict(self):
        pass

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
        print("Random Forest Regressor Results:")
        print("MSE:", self.mse)
        print("RMSE:", self.rmse)
        print("MAE:", self.mae)
        print("R²:", self.r2)

    def plot_graphs(self):
        residuos = self.y_test - self.y_pred.reshape(-1)
        plt.scatter(self.y_pred, residuos)
        plt.xlabel("Previsões")
        plt.ylabel("Resíduos")
        plt.title("Gráfico de Dispersão de Resíduos vs. Previsões")
        plt.show()

        sns.histplot(residuos, bins=30, kde=True)
        plt.xlabel("Resíduos")
        plt.ylabel("Frequência")
        plt.title("Histograma dos Resíduos")
        plt.show()

        stats.probplot(residuos, dist="norm", plot=plt)
        plt.title("Gráfico QQ dos Resíduos")
        plt.show()

    def run(self):
        self.preprocess_data()
        self.train_model()
        self.predict()
        self.evaluate()
        self.print_results()
        self.plot_graphs()
