import numpy as np 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split , GridSearchCV
from model_registry import Models_registry
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

def x_and_y(df , Target):
    X = df.drop(Target , axis=1)
    y = df[Target]

    X_train ,X_test ,y_train ,y_test = train_test_split(X ,y , test_size=0.2 , random_state=42 , shuffle=True) 

    return X_train , X_test , y_train, y_test

def grid_search(X_train ,X_test ,y_train ,y_test ,y_pred):
    models = Models_registry.get_models()
    for name , confing in models.items():
        

def fit_models(X_train , y_train):
    models = Models_registry.get_models()

def grid_search(X_train ,X_test ,y_train ,y_test ,y_pred):

