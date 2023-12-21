import xgboost as xgb

from regressor_base import RegressorBase


class XGBoostRegressorWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(file_path, output_filename="./output_files/xgb_model.pkl")

    def train_model(self):
        self.model = xgb.XGBRegressor(objective="reg:squarederror")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
