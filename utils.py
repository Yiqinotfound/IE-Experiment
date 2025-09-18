import numpy as np

def load_data(path:str):
    y_names = ["RM", "IMPACT_RST_AVE", "DWTT_AVE"]
    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}
    for y_name in y_names:
        X_train = np.load(f"data/{y_name}/X_train.npy")
        X_test = np.load(f"data/{y_name}/X_test.npy")
        y_train = np.load(f"data/{y_name}/y_train.npy")
        y_test = np.load(f"data/{y_name}/y_test.npy")
        X_train_dict[y_name] = X_train
        X_test_dict[y_name] = X_test
        y_train_dict[y_name] = y_train
        y_test_dict[y_name] = y_test
    return X_train_dict, X_test_dict, y_train_dict, y_test_dict
