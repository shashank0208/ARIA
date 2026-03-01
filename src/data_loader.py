# src/data_loader.py

import pandas as pd
from src.config import TRAIN_PATH, TEST_CLEAN_PATH, TEST_DRIFTED_PATH

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test_clean = pd.read_csv(TEST_CLEAN_PATH)
    test_drifted = pd.read_csv(TEST_DRIFTED_PATH)
    return train, test_clean, test_drifted