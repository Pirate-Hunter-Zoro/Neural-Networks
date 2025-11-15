import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import multiprocessing
from src.cnn.data import create_dataloaders
from src.cnn.models import Model1A, Model1B, Model2, Model3, Model4
from src.cnn.engine import train_step, test_step
from src.cnn.utils import plot_loss_and_accuracy

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model: nn.Module, model_name: str, dataset_name: str, loss_fn: nn.Module):
    train_dataloader, test_dataloader = create_dataloaders(dataset_name=dataset_name, batch_size=BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=DEVICE)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=DEVICE)
        print(f"{model_name} on {dataset_name}: Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        results["test_acc"].append(test_acc)
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
    
    # Plot results for model
    plot_loss_and_accuracy(results=results, filename=f"{dataset_name}_{model_name}_results.png")

def init_worker(loss_fn):
    global eval_model_with_args
    def eval_model_with_args(model_and_name_and_dataset: Tuple[nn.Module, str]):
        # This is dumb but its so that I can make the multiprocessing call below easier
        evaluate_model(model=model_and_name_and_dataset[0], model_name=model_and_name_and_dataset[1], dataset_name=model_and_name_and_dataset[2], loss_fn=loss_fn)
        
def train(model_and_name_and_dataset: Tuple[nn.Module, str]):
    global eval_model_with_args
    eval_model_with_args(model_and_name_and_dataset)

def main():
    loss_fn = nn.CrossEntropyLoss()

    model_1a = Model1A(num_classes=10).to(DEVICE)
    model_1b = Model1B(num_classes=10).to(DEVICE)
    model_2 = Model2(num_classes=10).to(DEVICE)
    model_3 = Model3(num_classes=10).to(DEVICE)
    model_4 = Model4(num_classes=10).to(DEVICE)
    
    models = [model_1a, model_1b, model_2, model_3, model_4]
    model_names = ["model_1a", "model_1b", "model_2", "model_3", "model_4"]
    dataset_names = ["fashion_mnist", "digits"]
    train_args = []
    for model in models:
        for model_name in model_names:
            for dataset_name in dataset_names:
                train_args.append((model, model_name, dataset_name))
    
    with multiprocessing.Pool(processes=len(models), initializer=init_worker, initargs=[loss_fn]) as pool:
        pool.map(train, train_args)

if __name__=="__main__":
    main()