from xg_boost_regressor_wrapper import XGBoostRegressorWrapper
from k_nearest_neighbors import KNNRegressorWrapper
from multi_layer_perceptron_wrapper import MultiLayerPerceptronWrapper
from random_forest_regressor_wrapper import RandomForestRegressorWrapper


FILE_PATH = "../dataset-extractor/output_files/materials_1st_group.csv"

# SOL,40,0.529
# HUME,2,0.04
# ADDI,115,0.08
# GELI,18,0.005
# CGRA,17,0.06
# CGRA,63,0.01
# HUIL,156,0.01
# EMUL,24,0.07
# CGRA,45,0.04
# HUIL,95,0.03
# HUIL,4,0.04
# HUIL,143,0.03
# CONS,63,0.011
# CONS,92,0.003
# EXTR,763,0.01
# EXTR,265,0.01
# ACTI,223,0.02
# PARF,345,0.002


def main():
    acronyms_list = [
        "SOL",
        "HUME",
        "ADDI",
        "GELI",
        "CGRA",
        "CGRA",
        "HUIL",
        "EMUL",
        "CGRA",
        "HUIL",
        "HUIL",
        "HUIL",
        "CONS",
        "CONS",
        "EXTR",
        "EXTR",
        "ACTI",
        "PARF",
    ]
    codes_list = [
        40,
        2,
        115,
        18,
        17,
        63,
        156,
        24,
        45,
        95,
        4,
        143,
        63,
        92,
        763,
        265,
        223,
        345,
    ]

    print()
    print("*** Starting the Random Forest Regressor algorithm ***")
    rf_model = RandomForestRegressorWrapper(FILE_PATH)
    rf_model.run()
    rf_model.predict_multiple(acronyms_list, codes_list)

    print()
    print("*** Starting the XG Boost Regressor algorithm ***")
    gb_model = XGBoostRegressorWrapper(FILE_PATH)
    gb_model.run()
    gb_model.predict_multiple(acronyms_list, codes_list)

    print()
    print("*** Starting the k-Nearest Neighbors algorithm ***")
    knn_model = KNNRegressorWrapper(FILE_PATH)
    knn_model.run()
    knn_model.predict_multiple(acronyms_list, codes_list)

    print()
    print("*** Starting the Multi-Layer Perceptron algorithm ***")
    mlp_model = MultiLayerPerceptronWrapper(FILE_PATH)
    mlp_model.run()
    mlp_model.predict_multiple(acronyms_list, codes_list)


if __name__ == "__main__":
    main()
