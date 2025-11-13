import torch.nn as nn

class Model1A(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            # You'll have to guess let the code crash once to see which dimension you need for the in features of the first liinear layer
            nn.Linear(in_features=1568, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    def forward(self, x):
        return self.layers(x)
        
        
class Model1B(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            # Gotta let the code crash to find out what in_features needs to be
            nn.Linear(in_features=6272, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    
class Model2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            # Gotta let the code crash to find out what in_features needs to be
            nn.Linear(in_features=6272, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)