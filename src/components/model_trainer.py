import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split # Make sure to import this for your data split outside this class

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_models



@dataclass
class ModelTraningConfig:
    """
    Configuration for model training paths and parameters.
    """
    traning_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    Class for training machine learning models.
    """
    def __init__(self):
        self.model_traning_config = ModelTraningConfig()

    def initiate_model_traning(self, train_array, test_array):
        """
        This method trains multiple regression models and evaluates their performance.
        It returns the best model based on R2 score.
        """
        try:
            logging.info("Entered the model training method or component.")

            logging.info("Splitting training and testing input and target variables.")
            # Splitting the train and test arrays into features and target variables
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            '''
            Those also work, but they are not used in the current code
            # If you want to use the train_test_split function, uncomment the following lines
                X_train, y_train, X_test, y_test = (
                    train_array[:, :-1],
                    train_array[:, -1],
                    test_array[:, :-1],
                    test_array[:, -1]
                )
            '''


            # creating models dictionary with various regression models
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regression' : DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'XGB Regressor': XGBRegressor(random_state=42),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False, random_state=42),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
            }

            # Define parameter grids for each model
            # IMPORTANT: Adjust these parameters and ranges based on your dataset and computational resources.
            # Start with broader ranges and narrow down as you see results.
            params = {
                'Linear Regression': {}, # No hyperparameters for basic Linear Regression in this context
                'Decision Tree Regression': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Random Forest Regressor': {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting Regressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7]
                },
                'XGB Regressor': {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2]
                },
                'KNeighbors Regressor': {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute']
                },
                'CatBoost Regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            # Call the updated evalute_models function
            model_report, tuned_models = evalute_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # Finding the best model based on R2 score from the model report dictionary
            best_model_score = max(sorted(model_report.values()))

            # Finding the best model name from the model report dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Retrieve the actual best tuned model instance
            best_model = tuned_models[best_model_name]

            # If the best model score is less than 0.6, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            save_object(
                file_path = self.model_traning_config.traning_model_file_path,
                obj = best_model
            )

            # generating predictions using the best model and calculating R2 score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            