from sklearn.svm import SVR

from regressor_base import RegressorBase


class SVRWrapper(RegressorBase):
    def train_model(self):
        self.model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred
