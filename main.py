from src.data import load_dataset
from src.network import NeuralNetwork
from src.utils import plot_loss, calculate_accuracy
import numpy as np
import json
from pathlib import Path

# Hyperparameters
LEARNING_RATE = 0.01
REG_STRENGTH = 0.0001
EPOCHS = 100

DATASETS = ['digits', 'credit', 'mushroom']

CONFIGURATIONS = {
    'first': [64, 64],
    'second': [64, 64, 64, 64],
    'third': [256, 256]
}

def main():
    accuracy_report = {}
    for dataset_name in DATASETS:
        print(f"--- Processing Dataset: {dataset_name} ---")
        accuracy_report[dataset_name] = {}
        X_train, y_train, X_test, y_test = load_dataset(dataset_name=dataset_name)
        input_dim = X_train.shape[1] # number of 'columns', per se
        num_classes = len(np.unique(y_train)) # we shall hope that the training set has all possible classes present at least once
        # Now try all the different configs on this dataset
        for config_name, config in CONFIGURATIONS.items():
            print(f"\nTraining {config_name} network with network size {config}...")
            network = NeuralNetwork(layer_dims=[input_dim] + config + [num_classes], reg_strength=REG_STRENGTH, learning_rate=LEARNING_RATE)
            train_losses, test_losses = network.train(X_train, y_train, X_test, y_test, EPOCHS)
            plot_file = f"{dataset_name}_{config_name}_loss.png"
            plot_loss(train_losses=train_losses, test_losses=test_losses, filename=plot_file)
            # Prediction results
            y_pred = network.predict(X_test)
            accuracy = calculate_accuracy(y_test, y_pred)
            print(f"Final Test Accuracy for {config_name}: {accuracy * 100:.2f}%")
            accuracy_report[dataset_name][config_name] = accuracy
    results_file = Path("outputs/accuracy_report.json")
    with open(results_file, 'w') as f:
        json.dump(accuracy_report, f, indent=4)
        print(f"Results stored in {str(results_file)}...")

if __name__ == "__main__":
    main()