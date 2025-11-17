# Machine Learning: Network Implementations

This project contains the source code for three distinct machine learning assignments, focusing on the implementation and analysis of core network architectures.

1. **Neural Network (from Scratch)**: A multi-layer perceptron built from the ground up using only `numpy`. It implements backpropagation, L2 regularization, and gradient descent.
2. **Convolutional Neural Network (PyTorch)**: A series of CNN models built in PyTorch to demonstrate the effects of filter count, Dropout, and Batch Normalization.
3. **Transformer (Attention)**: An implementation of Scaled Dot-Product Attention and Multi-Head Attention, verified by loading pre-trained weights from a BERT model.

## Project Structure

The project is organized into three main Python packages within the `src` directory, with a corresponding top-level script to run each experiment.

```text

.
├── src
│   ├── cnn/
│   ├── neural\_network/
│   └── transformer/
├── train\_cnn.py
├── train\_neural\_network.py
└── train\_attention.py

````

## Running the Experiments

Ensure all dependencies are installed from `requirements.txt` before running the scripts.

### Part 1: Neural Network (from Scratch)

This script trains a `numpy`-based neural network on the Digits, Credit, and Mushroom datasets.

**To Run:**

```bash
python -m train_neural_network
````

**Output:**
Loss plots and a final `accuracy_report.json` are saved to `src/neural_network/outputs/`.

### Part 2: Convolutional Neural Network (PyTorch)

This script trains five different CNN architectures (`Model1A`, `Model1B`, `Model2`, `Model3`, `Model4`) on both the Fashion-MNIST and MNIST Digits datasets.

**To Run:**

```bash
python -m train_cnn
```

**Output:**
Loss and accuracy plots for each model and dataset are saved to `src/cnn/outputs/`.

### Part 3: Transformer (Attention)

This script implements a Multi-Head Attention (MHA) block. It verifies the implementation by loading the weights from a pre-trained BERT model (Layers 0 and 11) and comparing the resulting attention maps.

**To Run:**

```bash
python -m run_attention
```

**Output:**
Attention head heatmaps are saved to `src/transformer/outputs/`.
