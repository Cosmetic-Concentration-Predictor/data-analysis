from sklearn.ensemble import RandomForestRegressor
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
