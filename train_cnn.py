import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from src.cnn.data import create_dataloaders
from src.cnn.models import Model1A, Model1B
from src.cnn.engine import train_step, test_step
from src.cnn.utils import plot_loss_and_accuracy

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model: nn.Module, model_name: str, loss_fn: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader):
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
        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        results["test_acc"].append(test_acc)
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
    
    # Plot results for first model
    plot_loss_and_accuracy(results=results, filename=f"{model_name}_results.png")

def main():
    train_dataloader, test_dataloader = create_dataloaders(dataset_name='fashion_mnist', batch_size=BATCH_SIZE)
    loss_fn = nn.CrossEntropyLoss()

    model_1a = Model1A(num_classes=10).to(DEVICE)
    model_1b = Model1B(num_classes=10).to(DEVICE)
    
    evaluate_model(model=model_1a, model_name="model_1a", loss_fn=loss_fn, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    evaluate_model(model=model_1b, model_name="model_1b", loss_fn=loss_fn, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    
if __name__=="__main__":
    main()