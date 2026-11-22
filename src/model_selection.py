import numpy as np 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import ElasticNet , Ridge , Lasso , LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from category_encoders import TargetEncoder

from preprcessing import data_loader, change_type, remove_outliers_and_log_price, df_dropna


def preprocess_data(df):
    df = change_type(df)
    df = df_dropna(df)
    return df


num_pip = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='mean')),
        ('scaler',MinMaxScaler())
    ]
)

category_pip = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('cat',TargetEncoder())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pip, ['Area']),
        ('category', category_pip, ['Address'])
    ]
)

class ModelRegistry:

    @staticmethod
    def get_models():

        return {

            "random_forest": {
                "estimator": RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=1
                ),
                "scale": False
            },

            "decision_tree": {
                "estimator": DecisionTreeRegressor(
                    random_state=42
                ),
                "scale": False
            },

            "svr": {
                "estimator": SVR(),
                "scale": True
            },

            "knn": {
                "estimator": KNeighborsRegressor(
                    n_neighbors=5
                ),
                "scale": True
            },

            "xgboost": {
                "estimator": XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    n_jobs=1
                ),
                "scale": False
            }
        }

