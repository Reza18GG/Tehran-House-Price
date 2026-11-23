from sklearn.linear_model import LinearRegression , Lasso , Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

class Models_registry:
    @staticmethod
    def get_models():
        return{
            'linear_regresion':{
                'model':LinearRegression()
            },

            'Lasso':{
                'model':Lasso(max_iter=10000)
            },

            'ridge':{
                'model': Ridge(max_iter=10000)
            },

            'decission_tree':{
                'model':DecisionTreeRegressor(max_depth=120,
                                                     max_leaf_nodes=10,
                                                     )
            },

            'ranadom_forest':{
                'model':RandomForestRegressor(max_depth=100,
                                                     n_estimators=200)
            },

            'KNN':{
                'model':KNeighborsRegressor(n_neighbors=3)
            },

            'SVM':{
                'model':SVR(kernel='linear',coef0=0.01)
            },
            'XGBOOST':{
                'model': XGBRegressor()
            }
        }

models = Models_registry.get_models()
print(models.keys())