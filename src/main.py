from gradient_boosting_regressor_wrapper import GradientBoostingRegressorWrapper
from k_nearest_neighbors import KNNRegressorWrapper
from multi_layer_perceptron_wrapper import MultiLayerPerceptronWrapper
from random_forest_regressor_wrapper import RandomForestRegressorWrapper

# from svr_wrapper import SVRWrapper

FILE_PATH = "../dataset-extractor/output_files/materials.csv"


def main():
    print()
    print("*** Starting the Random Forest Regressor algorithm ***")
    rf_model = RandomForestRegressorWrapper(FILE_PATH)
    rf_model.run()

    print()
    print("*** Starting the Gradient Boosting Regressor algorithm ***")
    gb_model = GradientBoostingRegressorWrapper(FILE_PATH)
    gb_model.run()

    print()
    print("*** Starting the k-Nearest Neighbors algorithm ***")
    knn_model = KNNRegressorWrapper(FILE_PATH)
    knn_model.run()

    print()
    print("*** Starting the Multi-Layer Perceptron algorithm ***")
    mlp_model = MultiLayerPerceptronWrapper(FILE_PATH)
    mlp_model.run()

    # print()
    # print("*** Starting the SVR algorithm ***")
    # svr_model = SVRWrapper(FILE_PATH, 50000)
    # svr_model.run()


if __name__ == "__main__":
    main()
