import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MUSHROOM_COLUMN_NAMES = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

RANDOM_STATE = 42

def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the dataset corresponding to the input name

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    if dataset_name == "mushroom":
        df = pd.read_csv(Path("data/mushroom/agaricus-lepiota.data"), header=None, names=MUSHROOM_COLUMN_NAMES, na_values="?")
        df = df.dropna()
        y = df['class']
        X = df.drop('class', axis=1)
        # Map the categories to numbers
        y = y.map({'p': 1, 'e': 0})
        # Turn the features into one-hot encodings - so instead of [f, g, f, y, s], you would have [[1 0 0 0] [0 1 0 0] [1 0 0 0] [0 0 1 0] [0 0 0 1]]
        X = pd.get_dummies(X)
        # Turn to numpy array
        X, y = X.to_numpy(), y.to_numpy()
    elif dataset_name == "credit":
        df = pd.read_csv(Path("data/credit.csv"))
        X, y = df.drop('Class', axis=1), df['Class']
        X, y = X.to_numpy(), y.to_numpy()
    elif dataset_name == "digits":
        digits = load_digits()
        X, y = digits.data, digits.target
    else:
        raise ValueError(f"Invalid dataset name provided: {dataset_name}...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    if dataset_name != "mushroom":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test