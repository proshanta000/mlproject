import os 
import sys
import pandas as pd
import numpy as np
import dill

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # Import these for tuning



def save_object(obj, file_path):
    """
    Save an object to a file using pickle.
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    


def evalute_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple machine learning models with hyperparameter tuning.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training target.
        X_test (np.array): Testing features.
        y_test (np.array): Testing target.
        models (dict): A dictionary of model instances (e.g., {'Linear Regression': LinearRegression()}).
        params (dict): A dictionary of hyperparameter grids/distributions for each model,
                       keyed by model name.

    Returns:
        tuple: A tuple containing:
               - dict: A dictionary containing the R2 score for each best-tuned model on the test set.
               - dict: A dictionary containing the best model instance (after tuning) for each model name.
    """
    try:
        report = {}
        tuned_models = {} # To store the best model instance after tuning

        for model_name, model in models.items():
            param_grid = params.get(model_name, {}) # Get parameters for the current model, default to empty dict

            logging.info(f"Starting evaluation/tuning for {model_name}...")

            if param_grid: # If parameters are defined, perform tuning
                logging.info(f"Performing GridSearchCV for {model_name} with parameters: {param_grid}")
                # Use GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3, # Number of cross-validation folds. Adjust based on dataset size and computational resources.
                    n_jobs=-1, # Use all available CPU cores for parallel processing
                    verbose=0, # Set to 1 or 2 for more detailed output during search
                    scoring='r2' # Use R2 score as the evaluation metric for regression
                )
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                logging.info(f"Best parameters for {model_name}: {best_params}")
                logging.info(f"Best cross-validation R2 score for {model_name}: {grid_search.best_score_:.4f}")
            else:
                logging.info(f"No specific hyperparameters defined for {model_name}. Training with default parameters.")
                model.fit(X_train, y_train)
                best_model = model # Use the default model if no params for tuning

            # Store the best tuned model (or default model if no tuning was done)
            tuned_models[model_name] = best_model

            # Predict on the training and testing data using the best model
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate R2 scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            # Store the model name and its R2 score (test score) in the report dictionary
            report[model_name] = test_score

            logging.info(f"{model_name} - Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

        return report, tuned_models # Return both the scores and the tuned models

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle.
    
    Args:
        file_path (str): Path to the file from which the object is to be loaded.
    
    Returns:
        obj: The loaded object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)