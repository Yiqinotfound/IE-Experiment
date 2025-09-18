import pandas as pd
import logging
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor


class PipelineSteel:
    def __init__(self, data_path, feature_selection:True, corr_threshold=0.9, vif_threshold=10):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold
        self.y_col_name = ["RM", "IMPACT_RST_AVE", "DWTT_AVE"]
        self.categorical_columns = ["OPERATING_MODE", "STEELGRADE", "STEEL_GRD_DETAIL"]
        self.time_cols = [
            "MILL_STA_DATE",
            "MILL_END_DATE",
            "START_COOLING",
            "STOP_COOLING",
        ]
        self.data_path = data_path
        # load dataframe and make sure time cols are string type
        self.origin_df = pd.read_csv(
            data_path, index_col=0, dtype={col: str for col in self.time_cols}
        )

        # deal with missing values
        self.X_df, self.y_df, self.new_df = self._drop_null()

        # devide data according to different y columns
        self.X_dict, self.y_dict = self._devide_data()

        # deal with time columns in X_dict.items()
        for key, X in self.X_dict.items():
            self.X_dict[key] = self._time_to_duration(df=X)
        logging.info(f"Time columns {str(self.time_cols)} converted to duration.")

        self.X_columns = self.X_dict["RM"].columns.tolist()

        # get dummies for categorical columns in X_dict.items()
        for key, X in self.X_dict.items():
            self.X_dict[key] = self._create_dummies(df=X)

        # save the dummy columns which starts with categorical columns
        self.dummy_columns = [
            col
            for col in self.X_dict["RM"].columns
            if any(col.startswith(cat_col) for cat_col in self.categorical_columns)
        ]
        logging.info(f"Dummy columns created: {self.dummy_columns}")
        self.X_columns = self.X_dict["RM"].columns.tolist()

        self.numerical_columns = [
            col for col in self.X_columns if col not in self.dummy_columns
        ]

        # convert all data to float
        for key, X in self.X_dict.items():
            self.X_dict[key] = X.astype(float)

        # scale the numerical columns
        for key, X in self.X_dict.items():
            self.X_dict[key] = self._scale_cols(X, self.numerical_columns)
        logging.info("Numerical columns scaled")

        # detect outliers and remove them
        for key, X in self.X_dict.items():
            outlier_mask = self._detect_outliers(
                X, cols=self.numerical_columns, threshold=3.0
            )
            self.X_dict[key] = self.X_dict[key][~outlier_mask]
            self.y_dict[key] = self.y_dict[key][~outlier_mask]

        # feature selection using correlation and VIF
        if feature_selection:
            logging.info("Starting feature selection using correlation and VIF.")
            logging.info(
                f"corr_threshold: {self.corr_threshold}, vif_threshold: {self.vif_threshold}"
            )
            for key in self.X_dict.keys():
                X_selected, selected_features = self.feature_selection_vif(
                    X=self.X_dict[key],
                    y=self.y_dict[key],
                    corr_threshold=self.corr_threshold,
                    vif_threshold=self.vif_threshold,
                )
                self.X_dict[key] = X_selected
                logging.info(
                    f"After feature selection for {key}, selected features: {selected_features}"
                )
                self.X_columns = self.X_dict[key].columns.tolist()
                logging.info(f"Final feature columns for {key}: {self.X_columns}")

        # store as numpy arrays
        self.X_np_dict = {key: X.values for key, X in self.X_dict.items()}
        self.y_np_dict = {key: y.values for key, y in self.y_dict.items()}
        logging.info("Data loaded and preprocessed.")

        self.N_RM = len(self.y_dict["RM"])
        self.d_RM = self.X_dict["RM"].shape[1]
        self.N_IMPACT = len(self.y_dict["IMPACT_RST_AVE"])
        self.d_IMPACT = self.X_dict["IMPACT_RST_AVE"].shape[1]
        self.N_DWTT = len(self.y_dict["DWTT_AVE"])
        self.d_DWTT = self.X_dict["DWTT_AVE"].shape[1]
        
        logging.info(
            f"N_RM: {self.N_RM}, N_IMPACT: {self.N_IMPACT}, N_DWTT: {self.N_DWTT}"
        )

        self.split_train_test(test_size=0.2, random_state=42)
        logging.info("Train-test split done.")
        
        
        # save data to numpy files
        for key in self.X_np_dict.keys():
            np.save(f"data/{key}/X_train.npy", self.X_train_dict[key])
            np.save(f"data/{key}/X_test.npy", self.X_test_dict[key])
            np.save(f"data/{key}/y_train.npy", self.y_train_dict[key])
            np.save(f"data/{key}/y_test.npy", self.y_test_dict[key])
        logging.info("Data saved to numpy files.")

    def _drop_null(self):
        # drop SLAB_NO and SMP_NO columns
        columns_to_drop = [
            col for col in self.origin_df.columns if col.startswith("SLAB_NO")
        ] + [col for col in self.origin_df.columns if col.startswith("SMP_NO")]
        new_df = self.origin_df.drop(columns=columns_to_drop)

        # find columns with more than 50% missing values
        self.missing_threshold = 0.5
        missing_ratio = new_df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index

        # do not drop y columns
        columns_to_drop = [col for col in columns_to_drop if col not in self.y_col_name]
        logging.info(
            f"Dropping columns with more than {self.missing_threshold*100}% missing values: {columns_to_drop}"
        )

        # drop columns with more than 50% missing values
        new_df = new_df.drop(columns=columns_to_drop)

        # find X columns
        self.X_columns = [col for col in new_df.columns if col not in self.y_col_name]

        X_df = new_df[self.X_columns]
        y_df = new_df[self.y_col_name]

        self.X_columns = X_df.columns.tolist()

        return X_df, y_df, new_df

    def _devide_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        X_dict, y_dict = {}, {}
        for y_col in self.y_col_name:
            subset = self.new_df[self.X_columns + [y_col]].dropna(
                subset=self.X_columns + [y_col]
            )
            X_dict[y_col] = subset[self.X_columns]
            y_dict[y_col] = subset[y_col]

        return X_dict, y_dict

    def _time_to_duration(self, df: pd.DataFrame):
        df["MILL_STA_DATE"] = pd.to_datetime(df["MILL_STA_DATE"])
        df["MILL_END_DATE"] = pd.to_datetime(df["MILL_END_DATE"])
        df["START_COOLING"] = pd.to_datetime(df["START_COOLING"], format="%Y%m%d%H%M%S")
        df["STOP_COOLING"] = pd.to_datetime(df["STOP_COOLING"], format="%Y%m%d%H%M%S")

        df["MILL_TIME"] = (df["MILL_END_DATE"] - df["MILL_STA_DATE"]).dt.total_seconds()
        df["COOLING_TIME"] = (
            df["STOP_COOLING"] - df["START_COOLING"]
        ).dt.total_seconds()
        df = df.drop(columns=self.time_cols)
        return df

    def _create_dummies(self, df: pd.DataFrame):
        df = df.copy()
        categorical_cols = df.columns.intersection(self.categorical_columns)

        # create dummy variables for categorical columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df

    def _scale_cols(self, df: pd.DataFrame, cols: list):
        """
        Scale the numerical cols using min-max scaling.
        """
        df = df.copy()
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val > 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0  # If all values are the same, set to 0.0
        return df

    def _detect_outliers(self, df: pd.DataFrame, cols: list, threshold=3.0):
        """
        Detect outliers in the specified columns using Z-score method.
        return outlier mask
        """

        df = df.copy()
        z_scores = np.abs(stats.zscore(df[cols], nan_policy="omit"))
        outlier_mask = (z_scores > threshold).any(axis=1)
        return outlier_mask

    def feature_selection_vif(
        self, X: pd.DataFrame, y: pd.Series, corr_threshold=0.9, vif_threshold=10
    ):
        X_selected = X.copy()

        zero_var_cols = X.columns[X.var() == 0].tolist()
        if zero_var_cols:
            logging.info(f"Removed zero-variance columns: {zero_var_cols}")
            X.drop(columns=zero_var_cols, inplace=True)

        corr_matrix = X_selected.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [
            column for column in upper.columns if any(upper[column] > corr_threshold)
        ]
        X_selected.drop(columns=to_drop_corr, inplace=True)

        vif_data = pd.DataFrame()
        vif_data["feature"] = X_selected.columns
        vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i)
                        for i in range(X_selected.shape[1])]
        high_vif_cols = vif_data.loc[vif_data["VIF"] > vif_threshold, "feature"].tolist()
        if high_vif_cols:
            print(f"Removed extremely high VIF columns: {high_vif_cols}")
            X_selected.drop(columns=high_vif_cols, inplace=True)

        # 3. 互信息筛选 (可选：根据互信息选择 top k 特征)
        mi = mutual_info_regression(X_selected, y)
        mi_series = pd.Series(mi, index=X_selected.columns)
        selected_features = mi_series.sort_values(ascending=False).index.tolist()

        return X_selected[selected_features], selected_features

    def split_train_test(self, test_size=0.2, random_state=42):
        self.X_train_dict = {}
        self.X_test_dict = {}
        self.y_train_dict = {}
        self.y_test_dict = {}

        for key in self.X_np_dict.keys():
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_np_dict[key],
                self.y_np_dict[key],
                test_size=test_size,
                random_state=random_state,
            )
            self.X_train_dict[key] = X_train
            self.X_test_dict[key] = X_test
            self.y_train_dict[key] = y_train
            self.y_test_dict[key] = y_test
