# MNIST Handwritten Digit Classifier: A Neural Network Built from Scratch

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-91.81%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A two-layer neural network that recognizes handwritten digits (0 to 9), implemented using **only NumPy**. There is no TensorFlow, PyTorch, or scikit-learn: forward propagation, backpropagation, and gradient descent are all written by hand so every step of learning is visible.

The model is trained on 59,000 images, tuned against a 1,000-image dev split, and finally evaluated on the **10,000 unseen images** in `mnist_test.csv`.

> **Why this project?** Frameworks hide the math. This repo opens the box and shows how a network turns pixels into predictions and how it corrects its own mistakes.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [The Math](#the-math)
- [Results](#results)
- [Evaluation on the Test Set](#evaluation-on-the-test-set)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Code Walkthrough](#code-walkthrough)
- [Design Decisions](#design-decisions)
- [Limitations and Future Work](#limitations-and-future-work)
- [Key Learnings](#key-learnings)
- [License](#license)

---

## How It Works

1. **Load** 28×28 grayscale images, each flattened into 784 pixel values.
2. **Split** the training file into 59,000 training and 1,000 dev images.
3. **Normalize** pixels from 0–255 to 0–1.
4. **Forward pass:** pixels flow through a hidden layer (ReLU) and an output layer (softmax) to produce 10 probabilities, one per digit.
5. **Backward pass:** compare the probabilities against the true label and compute how much each weight contributed to the error.
6. **Update** every weight a small step in the direction that reduces the error. Repeat for 1,000 iterations.
7. **Evaluate** on the dev set and, once, on the untouched test set.

---

## Architecture

```
 Input layer           Hidden layer            Output layer
 (784 pixels)          (10 neurons, ReLU)      (10 neurons, Softmax)

  x1  ──┐
  x2  ──┤                 h1 ──┐                  P(digit 0)
  x3  ──┼──►  W1, b1 ──►  h2 ──┼──►  W2, b2 ──►   P(digit 1)
  ...   │                 ...  │                  ...
  x784 ─┘                 h10 ─┘                  P(digit 9)
```

| Layer  | Shape        | Activation | Parameters                |
|--------|--------------|------------|---------------------------|
| Input  | 784 × m      | none       | none                      |
| Hidden | W1: 10 × 784 | ReLU       | 7,840 weights + 10 biases |
| Output | W2: 10 × 10  | Softmax    | 100 weights + 10 biases   |

**Total trainable parameters: 7,960**, where *m* is the number of examples.

---

## The Math

**Forward propagation**

```
Z1 = W1 · X + b1
A1 = ReLU(Z1) = max(0, Z1)
Z2 = W2 · A1 + b2
A2 = softmax(Z2)
```

**Backpropagation** (softmax with cross-entropy gives a clean gradient)

```
dZ2 = A2 − one_hot(Y)
dW2 = (1/m) · dZ2 · A1ᵀ
db2 = (1/m) · Σ dZ2
dZ1 = W2ᵀ · dZ2 ⊙ ReLU′(Z1)
dW1 = (1/m) · dZ1 · Xᵀ
db1 = (1/m) · Σ dZ1
```

**Gradient descent update** (learning rate α = 0.1)

```
W ← W − α · dW
b ← b − α · db
```

---

## Results

### Training progress

Training for 1,000 iterations with a learning rate of 0.1 on 59,000 images:

| Iteration | Training Accuracy |
|-----------|-------------------|
| 0         | 13.52%            |
| 100       | 83.49%            |
| 200       | 87.01%            |
| 300       | 88.70%            |
| 400       | 89.63%            |
| 500       | 90.25%            |
| 600       | 90.68%            |
| 700       | 90.99%            |
| 800       | 91.27%            |
| 900       | 91.47%            |

At iteration 0 the network is guessing (about 10% is chance level). Accuracy jumps past 83% within the first 100 iterations, then improves with diminishing returns.

### Final accuracy

| Dataset  | Images | Accuracy |
|----------|--------|----------|
| Train    | 59,000 | 91.66%   |
| Dev      | 1,000  | 90.50%   |
| **Test** | 10,000 | **91.81%** |

The small gap between training and test accuracy shows the model generalizes to digits it has never seen rather than memorizing the training set.

---

## Evaluation on the Test Set

Evaluation is a single forward pass with no weight updates. The test images get the **same preprocessing** as training (transpose, then divide by 255):

```python
test_data = np.array(pd.read_csv("../datasets/mnist_test.csv")).T
Y_test = test_data[0]
X_test = test_data[1:] / 255.0

_, _, _, A2 = forward_prop(W1, b1, W2, b2, X_test)
test_preds = get_predictions(A2)
print(f"Test accuracy: {np.mean(test_preds == Y_test) * 100:.2f}%")
```

### Confusion matrix

Rows are the true digit and columns are the predicted digit. Off-diagonal cells show which digits the model confuses.

![Confusion matrix](images/confusion_matrix.png)

### Misclassified examples

![Misclassified digits](images/misclassified.png)

Typical mistakes involve digits that look alike when handwritten, such as 4 and 9, 3 and 5, or 7 and 9.

---

## Project Structure

```
MNSIT_Neural_Network_Model/
├── datasets/
│   ├── mnist_train.csv      # 60,000 labeled images (stored with Git LFS)
│   └── mnist_test.csv       # 10,000 labeled images
├── images/
│   ├── confusion_matrix.png # Test-set confusion matrix
│   └── misclassified.png    # Examples the model got wrong
├── model/
│   └── mnsit_model.ipynb    # Data prep, training, and evaluation
├── LICENSE                  # MIT
└── README.md
```

**Dataset format:** each CSV row is `label, pixel1, pixel2, ..., pixel784`. The first column is the digit (0–9) and the other 784 columns are pixel intensities (0–255).

---

## Getting Started

### Prerequisites

- Python 3.9 or newer (developed on 3.12)
- [Git LFS](https://git-lfs.com) (the training CSV is about 110 MB)

### Installation

```bash
# 1. Install Git LFS once, then clone
git lfs install
git clone https://github.com/Kushal1706/MNSIT_Neural_Network_Model.git
cd MNSIT_Neural_Network_Model

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas matplotlib jupyter
```

> If `datasets/mnist_train.csv` is only a few lines of text starting with `version https://git-lfs...`, Git LFS was not active when you cloned. Run `git lfs pull` to download the real file.

### Run

```bash
jupyter notebook model/mnsit_model.ipynb
```

Run all cells from top to bottom. Training takes well under a minute on a typical laptop because the network is small and fully vectorized. Results vary slightly between runs because of the random shuffle and initialization; add `np.random.seed(42)` at the top of the notebook for repeatable numbers.

---

## Code Walkthrough

| Function | What it does |
|----------|--------------|
| `init_params()` | Creates the starting weights (He initialization) and zero biases |
| `ReLU(Z)` | Keeps positive values, zeroes out negatives |
| `softmax(Z)` | Converts 10 raw scores into probabilities that sum to 1 |
| `forward_prop(...)` | Pixels → hidden activations → output probabilities |
| `one_hot(Y)` | Turns label `3` into `[0,0,0,1,0,0,0,0,0,0]` |
| `deriv_ReLU(Z)` | Gradient of ReLU: 1 where Z > 0, otherwise 0 |
| `back_prop(...)` | Computes the gradient of every weight and bias |
| `update_params(...)` | Applies one gradient descent step |
| `get_predictions(A2)` | Picks the digit with the highest probability |
| `get_accuracy(...)` | Fraction of predictions that match the labels |
| `gradient_descent(...)` | Runs the full training loop and reports accuracy every 100 iterations |

**Data preparation:** the data is shuffled, then split into 1,000 dev examples and 59,000 training examples. It is transposed so that each *column* is one image, which makes the matrix math `W · X` line up cleanly.

---

## Design Decisions

- **He initialization** (`randn * sqrt(2 / fan_in)`): suited to ReLU layers, it keeps signals from shrinking or exploding at the start of training.
- **Numerically stable softmax:** subtracting the column maximum before `exp` prevents overflow without changing the result.
- **Pixel normalization (`/ 255`):** keeps inputs in a small range so gradient descent is stable.
- **Fully vectorized NumPy:** the whole dataset is processed with matrix operations, with no Python loops over images.
- **Full-batch gradient descent:** simple and deterministic, ideal for learning how the algorithm works.
- **Separate dev and test sets:** the dev set is for comparing choices like learning rate or hidden size. The test set is used only for the final reported number.

---

## Key Learnings

- A neural network is matrix multiplications plus non-linearities, nothing more mysterious.
- Softmax and cross-entropy pair together to give the simple gradient `A2 − Y`.
- Backpropagation applies the chain rule layer by layer.
- Training accuracy alone can mislead: a held-out test set is what shows real performance.
- Initialization, normalization, and learning rate all affect training stability.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author:** [Kushal](https://github.com/Kushal1706)

If this helped you understand neural networks, consider giving the repo a star.
