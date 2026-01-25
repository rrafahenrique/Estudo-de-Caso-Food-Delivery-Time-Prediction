import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb

def train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42        
):
    """
    Treina e avalia múltiplos modelos de Regressão,
    retornando as métricas de avaliação.
    """   

    #Modelos
    models: Dict[str, object] = {
        "Regressão Linear": LinearRegression(),
        "LightGBM": lgb.LGBMRegressor(random_state=random_state, verbosity=-1),
        "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
        "XGBoost": xgb.XGBRegressor(random_state=random_state, objective = 'reg:squarederror'),
        "CatBoost": cb.CatBoostRegressor(random_state=random_state, verbose=False)
    }

    results = []

    #Loop de treino e avaliação
    for name, model in models.items():
        model.fit(X_train, y_train) #treino
        y_pred = model.predict(X_test) # fit

        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        rmsle = mean_squared_log_error(y_test, y_pred)
        adj_r2 = 1 - ((1 - r2) * (len(y_test) - 1)) / (len(y_test) - X_train.shape[1] - 1)

        results.append({
            'Modelo': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'R² Ajustado': adj_r2,
            'MAPE (%)': mape,
            'RMSLE': rmsle    
        })

    #Dataframe Final
    results_df = pd.DataFrame(results)

    return results_df.round(3)

