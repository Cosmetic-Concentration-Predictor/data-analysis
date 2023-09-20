import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def run(self):
        self.preprocess_data()
        self.train_model()
        self.predict()
        self.evaluate()
        self.print_results()

    def print_results(self):
        pass
