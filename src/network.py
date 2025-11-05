import numpy as np
from typing import List, Tuple

from src.layers import LinearReLU, LinearCrossEntropy

class NeuralNetwork:
    
    def __init__(self, layer_dims: List[int], reg_strength: float, learning_rate: float):
        self.reg = reg_strength
        self.lr = learning_rate
        self.layers = [
                        LinearReLU(input_dim=layer_dims[i], output_dim=layer_dims[i+1]) 
                        if i < len(layer_dims)-2 else 
                        LinearCrossEntropy(input_dim=layer_dims[i], output_dim=layer_dims[i+1]) 
                        for i in range(len(layer_dims)-1)
                    ]
        
    def forward(self, X: np.ndarray, y: np.ndarray) -> float:
        """Forward pass of the function to receive an input and compute the resulting cross entropy loss given the output

        Args:
            X (np.ndarray): Input observations
            y (np.ndarray): Expected outputs

        Returns:
            float: Resulting cross entropy loss
        """
        A = X
        # Each layer stores what it needs to for when we call back propagation
        for layer in self.layers[:-1]:
            A = layer.forward(A)
        return self.layers[-1].forward(A, y, self.reg) # Returns the cross entropy loss of the final layer
    
    def backward(self):
        """Perform backward pass on all layers contained within network
        """
        # First compute the loss of the final activations passed into the cross entropy layer of our network
        # Each layer stores what it needs to for when we update
        dA = self.layers[-1].backward(self.reg)
        for i in range(len(self.layers)-2, -1, -1):
            dA = self.layers[i].backward(dA, self.reg)
        
    def update(self):
        """Have each linear forward layer of the network modify its weights according to its calculated weight loss
        """
        for layer in self.layers:
            layer.weights -= self.lr * layer.dW
            layer.biases -= self.lr * layer.db
            
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int) -> Tuple[List[float], List[float]]:
        """Train over the training data and report performance on the testing data

        Args:
            X_train (np.ndarray): Train observations
            y_train (np.ndarray): Train outputs
            X_test (np.ndarray): Test observations
            y_test (np.ndarray): Test outputs
            epochs (int): Number of epochs of training

        Returns:
            Tuple[List[float], List[float]]: Resulting training and testing losses
        """
        train_losses = []
        test_losses = []
        for _ in range(epochs):
            # Run a forward pass
            train_loss = self.forward(X_train, y_train)
            train_losses.append(train_loss)
            # Now we can train the network - this is exact gradient descent
            self.backward()
            self.update()
            test_loss = self.forward(X_test, y_test)
            test_losses.append(test_loss)
            
        return (train_losses, test_losses)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the classes of the given test observations

        Args:
            X (np.ndarray): Input test observations

        Returns:
            np.ndarray: Output predictions
        """
        # Partial forward pass for predicting accuracy
        A = X
        for linear_relu in self.layers[:-1]:
            A = linear_relu.forward(A)
        return self.layers[-1].predict(A)