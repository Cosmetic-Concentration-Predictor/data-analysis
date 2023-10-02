from sklearn.neighbors import KNeighborsRegressor

from regressor_base import RegressorBase


class KNNRegressorWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(file_path, output_filename="./output_files/knn_model.pkl")

    def train_model(self):
        self.model = KNeighborsRegressor()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
