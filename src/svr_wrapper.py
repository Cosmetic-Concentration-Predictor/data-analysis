import numpy as np
from sklearn.svm import SVR

from regressor_base import RegressorBase


class SVRWrapper(RegressorBase):
    def __init__(self, file_path):
        super().__init__(file_path, output_filename="./output_files/svr_model.pkl")

    def train_model(self):
        self.model = SVR(kernel="rbf", C=1.0, cache_size=200, epsilon=0.1)
        self._train_in_batches(100000)

    def _train_in_batches(self, batch_size=10000):
        for i in range(0, len(self.X_train), batch_size):
            print(f"Iteração n° {i}")
            X_batch = self.X_train[i : i + batch_size]
            y_batch = self.y_train[i : i + batch_size]

            self.model.fit(X_batch, y_batch)

            y_pred_batch = self.model.predict(self.X_test)

            if i == 0:
                self.y_pred = y_pred_batch
            else:
                self.y_pred = np.concatenate((self.y_pred, y_pred_batch))

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
