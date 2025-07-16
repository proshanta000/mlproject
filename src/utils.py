import os 
import sys
import pandas as pd
import numpy as np
import dill

from src.logger import logging
from src.exception import CustomException



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