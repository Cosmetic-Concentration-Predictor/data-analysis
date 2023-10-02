from sklearn.ensemble import GradientBoostingRegressor

from regressor_base import RegressorBase


class GradientBoostingRegressorWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(file_path, output_filename="./output_files/gb_model.pkl")

    def train_model(self):
        self.model = GradientBoostingRegressor(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
