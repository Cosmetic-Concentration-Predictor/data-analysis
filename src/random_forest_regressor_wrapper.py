from sklearn.ensemble import RandomForestRegressor

from regressor_base import RegressorBase


class RandomForestRegressorWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(
            file_path,
            output_model="./output_files/rf_model.joblib",
            output_encoder="./output_files/rf_encoder.joblib",
        )

    def train_model(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
