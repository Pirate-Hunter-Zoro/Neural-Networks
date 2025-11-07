import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

DEST_PATH = Path("src/cnn/outputs/")

def plot_loss_and_accuracy(results: Dict, filename: str):
    """Helper function to plot the included loss and accuracy

    Args:
        results (Dict): Containing loss and accuracy
        filename (str): Where to save the plot file
    """
    DEST_PATH.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(1, 2, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(results['train_loss'], label="train loss")
    plt.plot(results['test_loss'], label="test loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(results['train_acc'], label='train accuracy')
    plt.plot(results['test_acc'], label='test accuracy')
    plt.legend()
    plt.savefig(f"src/cnn/outputs/{filename}")
    plt.close()