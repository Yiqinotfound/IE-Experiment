from models.linear_model import LinearModel
from dataset.dataset import PipelineSteel
import numpy as np
from utils import load_data


def main():
    data_path = "data/data.csv"
    y_names = ["RM", "IMPACT_RST_AVE", "DWTT_AVE"]
    X_train_dict, X_test_dict, y_train_dict, y_test_dict = load_data(path=data_path)

    for y_name in y_names:
        X_train = X_train_dict[y_name]
        X_test = X_test_dict[y_name]
        y_train = y_train_dict[y_name]
        y_test = y_test_dict[y_name]

        model = LinearModel(0)
        model.fit(X_train=X_train, y_train=y_train)
        rmse_train, mae_train, r2_train = model.evaluate(X_train, y_train)
        rmse_test, mae_test, r2_test = model.evaluate(X_test, y_test)
        
        print(f"---------------Results for target: {y_name}---------------")

        print(f"RMSE Train: {rmse_train:.4f}")
        print(f"MAE Train: {mae_train:.4f}")
        print(f"R2 Train: {r2_train:.4f}")
        print(f"RMSE Test: {rmse_test:.4f}")
        print(f"MAE Test: {mae_test:.4f}")
        print(f"R2 Test: {r2_test:.4f}")


if __name__ == "__main__":
    main()
