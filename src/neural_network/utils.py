import matplotlib.pyplot as plt
from typing import List
import numpy as np

def plot_loss(train_losses: List[int], test_losses: List[int], filename: str):
    """Generate and save the plot of the performance

    Args:
        train_losses (List[int]): Training losses by the epoch
        test_losses (List[int]): Testing losses by the epoch
        filename (str): Name of the file to save the plotted results in
    """
    epochs = range(len(train_losses))
    plt.figure()
    plt.title("Loss by the Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Testing Loss")
    plt.legend()
    plt.savefig(f"src/neural_network/outputs/{filename}")
    plt.clf()
    
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Method to calculate the accuracy given the predicted classes and actual classes

    Args:
        y_true (np.ndarray): actual classes
        y_pred (np.ndarray): predicted classes

    Returns:
        float: resulting accuracy
    """
    return np.sum(y_pred == y_true) / len(y_true)