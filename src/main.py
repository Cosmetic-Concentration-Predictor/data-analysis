from xg_boost_regressor_wrapper import XGBoostRegressorWrapper
from k_nearest_neighbors import KNNRegressorWrapper
from multi_layer_perceptron_wrapper import MultiLayerPerceptronWrapper
from random_forest_regressor_wrapper import RandomForestRegressorWrapper


FILE_PATH = "../dataset-extractor/output_files/materials_2nd_group.csv"


def main():
    print()
    print("*** Starting the Random Forest Regressor algorithm ***")
    rf_model = RandomForestRegressorWrapper(FILE_PATH)
    rf_model.run()

    print()
    print("*** Starting the XG Boost Regressor algorithm ***")
    gb_model = XGBoostRegressorWrapper(FILE_PATH)
    gb_model.run()

    print()
    print("*** Starting the k-Nearest Neighbors algorithm ***")
    knn_model = KNNRegressorWrapper(FILE_PATH)
    knn_model.run()

    print()
    print("*** Starting the Multi-Layer Perceptron algorithm ***")
    mlp_model = MultiLayerPerceptronWrapper(FILE_PATH)
    mlp_model.run()


if __name__ == "__main__":
    main()
