import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from regressor_base import RegressorBase


class RandomForestRegressorWrapper(RegressorBase):
    def preprocess_data(self):
        self.X = self.data.drop("concentration", axis=1)
        self.y = self.data["concentration"]

        label_encoder = LabelEncoder()
        self.X["acronym"] = label_encoder.fit_transform(self.X["acronym"])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.model.fit(self.X_train, self.y_train)

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
        print("Random Forest Regressor Results:")
        print("MSE:", self.mse)
        print("RMSE:", self.rmse)
        print("MAE:", self.mae)
        print("R²:", self.r2)
