from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Tuple

def create_dataloaders(dataset_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Helper function to create the DataLoader objects associated with the specified dataset

    Args:
        dataset_name (str): Name of the data set
        batch_size (int): Batch size to create the DataLoader with

    Returns:
        Tuple[DataLoader, DataLoader]: Resulting DataLoaders for train and test data
    """
    transform = ToTensor()
    if dataset_name == "fashion_mnist":
        dataset_class = datasets.FashionMNIST
    elif dataset_name == "digits":
        dataset_class = datasets.MNIST
    else:
        raise ValueError(f"Invalid dataset specified: {dataset_name}...")
    data_train = dataset_class(train=True, download=True, transform=transform)
    data_test = dataset_class(train=False, download=True, transform=transform)
    
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size)
    return (train_loader, test_loader)