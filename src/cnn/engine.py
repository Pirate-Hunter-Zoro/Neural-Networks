import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> Tuple[float, float]:
    """Helper method to perform one epoch of training

    Args:
        model (nn.Module): model to train
        dataloader (DataLoader): data loader
        loss_fn (nn.Module): loss definition - gradient calculator
        optimizer (torch.optim.Optimizer): gradient stepper
        device (str): cpu or cuda

    Returns:
        Tuple[float, float]: Resulting training loss and training accuracy
    """
    # Put model in train mode
    model.train()
    train_loss = 0
    train_acc = 0
    for (X, y) in dataloader:
        X = X.to(device)
        y = y.to(device)
        # Get model predictions
        y_pred = model(X)
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # Zero the gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # Update model
        optimizer.step()
        
        # Get accuracy
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item()
        
    # Find average loss and accuracy
    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)
    
    return (train_loss, train_acc)

def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: str) -> Tuple[float, float]:
    """Helper method to perform one epoch of testing

    Args:
        model (nn.Module): model to train
        dataloader (DataLoader): data loader
        loss_fn (nn.Module): loss definition - gradient calculator
        device (str): cpu or cuda

    Returns:
        Tuple[float, float]: Resulting testing loss and testing accuracy
    """
    # Put model in eval mode
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        # No gradient tracking
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)
            # Get model predictions
            y_pred = model(X)
            # Calculate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            # Get accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item()
            
        # Find average loss and accuracy
        test_loss /= len(dataloader)
        test_acc /= len(dataloader.dataset)
    
    return (test_loss, test_acc)