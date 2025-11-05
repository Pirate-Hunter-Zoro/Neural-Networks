import numpy as np


class LinearReLU:
    
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.zeros(output_dim)
        # Cache needed for backward propagation
        self.cache = {}
        self.dW = None
        self.db = None
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward activation for this linear layer (while caching what it needs to)

        Args:
            X (np.ndarray): input observations

        Returns:
            np.ndarray: resulting output activation
        """
        Z = np.matmul(X, self.weights) + self.biases
        A = np.maximum(0, Z) # ReLU activation
        self.cache['X'] = X
        self.cache['Z'] = Z
        return A
    
    def backward(self, dA: np.ndarray, reg_strength: float) -> np.ndarray:
        """Perform backward propagation given the loss of the preceding activation layer

        Args:
            dA (np.ndarray): Loss of the activation layer
            reg_strength (float): For L2 regularization  

        Returns:
            np.ndarray: Resulting loss to pass on to the previous layer 
        """
        # Note that the dA input is denoted dL/dA in the comments below, and we will denote dL/dVAR as dVAR in code
        
        # Values needed for preceding derivatives
        X, Z = self.cache['X'], self.cache['Z']
        
        # dL/dZ = dL/dA * dA/dZ = dL/dA whenever Z did not become 0 with ReLU
        mask = np.ones(Z.shape)
        mask[Z <= 0] = 0
        dZ = dA * mask
        
        # dL/dW = dL/dZ * dZ/dW + reg_loss = dL/dZ * X + reg_strength*W
        dW = np.matmul(X.T, dZ) + reg_strength*self.weights
        
        # dL/db = dL/dZ * dZ/db = dL/dZ * 1 = dL/dZ, but we must concatenate this over the entire batch (matrix -> vector)
        db = np.sum(dZ, axis=0)
        
        # dL/dX = dL/dZ * dZ/dX = dL/dZ * W
        dX = np.matmul(dZ, self.weights.T) # (N, D_out) x (D_out, D_in) -> (N, D_in)
        
        # Store these for later use
        self.dW = dW
        self.db = db
        
        return dX
    
    
def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Given a bunch of labels, turn the vector into a one-hot encoding

    Args:
        y (np.ndarray): integer feature labels
        num_classes (int): total number of classes possible

    Returns:
        np.ndarray: corresponding one-hot encoded vector
    """
    y_one_hot = np.zeros(shape=(y.shape[0], num_classes))
    # Go through all observations in y, and for that corresponding vector in y_one_hot, make the index of that vector 1 at whatever value y took on
    y_one_hot[np.arange(y.shape[0]), y] = 1 
    return y_one_hot


class LinearCrossEntropy:
    
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.zeros(output_dim)
        self.cache = {}
        
    def forward(self, X: np.ndarray, y: np.ndarray, reg_strength: float) -> float:
        """Run the forward pass given inputs and return the resulting loss

        Args:
            X (np.ndarray): Activations from previous layer
            y (np.ndarray): Target values
            reg_strength (float): Regularization constant

        Returns:
            float: Resulting loss - combining data loss and regularization loss
        """
        Z = np.matmul(X, self.weights) + self.biases
        # Apply softmax with a stabilization trick
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True)) # Since exp is monotonic, as is the shift by the max Z value, this will still favor the highest Z value index
        probs = exps / np.sum(exps, axis=1, keepdims=True) # Softmax
        # Now we calculate the cross entropy loss
        y_one_hot = one_hot_encode(y, num_classes=self.weights.shape[1])
        # When taking log of probabilities, add a small scalar to prevent taking the log of 0
        data_loss = -np.mean(np.sum(y_one_hot*np.log(probs + 1e-9), axis=1))
        # As well as the regularization loss - 1/2 * lambda||W||^2
        reg_loss = 0.5 * reg_strength * np.sum(self.weights * self.weights)
        self.cache['X'] = X
        self.cache['y_one_hot'] = y_one_hot
        self.cache['probs'] = probs
        
        return data_loss + reg_loss
    
    def backward(self, reg_strength: float) -> np.ndarray:
        """Backward propagation to return the loss of the previous activations passed in - denoted 'X' in the cache

        Args:
            reg_strength (float): Regularization constant

        Returns:
            np.ndarray: Resulting loss of previous activations
        """
        X, y_one_hot, probs = self.cache['X'], self.cache['y_one_hot'], self.cache['probs']
        
        # Gradient for softmax with cross entropy simplifies to 'probs - y_one_hot' - we use this instead of having a dA like we did in the LinearReLU
        dZ = probs - y_one_hot
        
        # The rest of the loss values are identical in nature to what we did in LinearReLU
        # dL/dW = dL/dZ * dZ/dW + reg_loss = dL/dZ * X + reg_strength*W
        dW = np.matmul(X.T, dZ) + reg_strength*self.weights
        
        # dL/db = dL/dZ * dZ/db = dL/dZ * 1 = dL/dZ, but we must concatenate this over the entire batch (matrix -> vector)
        db = np.sum(dZ, axis=0)
        
        # dL/dX = dL/dZ * dZ/dX = dL/dZ * W
        dX = np.matmul(dZ, self.weights.T) # (N, D_out) x (D_out, D_in) -> (N, D_in)
        
        # Store these for later use
        self.dW = dW
        self.db = db
        
        return dX
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Helper method to perform the forward pass but instead of returning the loss, return the classifications

        Args:
            X (np.ndarray): Input observations (activations from previous layer of a network)

        Returns:
            np.ndarray: Resulting classification predictions
        """
        Z = np.matmul(X, self.weights) + self.biases
        # Just return the maximum classification index for each observation
        return np.argmax(Z, axis=1)