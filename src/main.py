from random_forest_regressor_wrapper import RandomForestRegressorWrapper
from svr_wrapper import SVRWrapper
from neural_network_wrapper import NeuralNetworkWrapper

# FILE_PATH = "../extract_dataset/materials.csv"
FILE_PATH = "../extract_dataset/src/materials.csv"


def main():
    print()
    print("*** Starting the Random Forest Regressor algorithm ***")
    rf_model = RandomForestRegressorWrapper(FILE_PATH)
    rf_model.run()

    print()
    print("*** Starting the Neural Network algorithm ***")
    svm_model = NeuralNetworkWrapper(FILE_PATH)
    svm_model.run()

    print()
    print("*** Starting the SVR algorithm ***")
    svm_model = SVRWrapper(FILE_PATH)
    svm_model.run()


if __name__ == "__main__":
    main()
