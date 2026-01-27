import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
import optuna

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import catboost as cb

import logging
# Opcional: Define o nível de log para CRITICAL para garantir que nada seja exibido
optuna.logging.set_verbosity(logging.CRITICAL)

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
#---------------------------------------------------------------------------------------
def regression_metrics(nome_modelo, y_test, y_pred, X_test):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return {
        "Modelo": nome_modelo,
        "MAE": mae,
        "MSE": mse,
        "R²": r2,
        "MAPE (%)": mape,
    }

#--------------------------------------------------------------------------------------------
def optuna_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    modelo="catboost",
    n_trials=50,
    random_state=42
):
    """
    modelo: 'catboost' ou 'lightgbm'
    """

    def objective(trial):

        if modelo == "catboost":
            params = {
                "loss_function": "MAE",
                "eval_metric": "MAE",

                "iterations": trial.suggest_int("iterations", 500, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),

                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),

                "random_strength": trial.suggest_float("random_strength", 0, 10),

                "verbose": False
            }

            model = cb.CatBoostRegressor(**params)

        elif modelo == "lightgbm":
            params = {
                "objective": "regression",
                "metric": "mae",

                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 4, 12),

                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),

                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),

                "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),

                "verbosity": -1,
            }

            model = lgb.LGBMRegressor(**params)

        else:
            raise ValueError("Modelo deve ser 'catboost' ou 'lightgbm'")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return mean_absolute_error(y_test, y_pred)

    # Estudo Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Melhor modelo
    best_params = study.best_params

    if modelo == "catboost":
        best_model = cb.CatBoostRegressor(
            **best_params,
            loss_function="MAE",
            verbose=False,
            random_seed=random_state
        )
        nome_modelo = "CatBoost"

    else:
        best_model = lgb.LGBMRegressor(
            **best_params,
            objective="regression",
            metric="mae",
            verbosity=-1,
            random_state=random_state
        )
        nome_modelo = "LightGBM"

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Métricas
    metrics = regression_metrics(
        nome_modelo, y_test, y_pred, X_test
    )

    return pd.DataFrame([metrics]), best_model, study

