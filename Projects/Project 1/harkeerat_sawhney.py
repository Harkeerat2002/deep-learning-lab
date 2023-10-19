"""
Assignment 1: Polynomial Regression
Student: Harkeerat Singh Sawhney
"""
# Libraries:
import matplotlib.pyplot as plt
import numpy as np
import platform
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
}  # Or palatino if you like it more
plt.rcParams.update(params)

# A test (Uncomment to see the result)
# plt.figure()
# plt.xlabel(r"This is a CMS caption with some math: $\int_0^\infty f(x) dx$")
# plt.show()


# *** Question 1 **
def plot_polynomial(coeffs, z_range, color="b"):
    print("Starting Exercise 1")
    w0, w1, w2, w3, w4 = coeffs
    z_min, z_max = z_range
    z = np.linspace(z_min, z_max, 100)
    y = w0 + w1 * z + w2 * z**2 + w3 * z**3 + w4 * z**4
    plt.rcParams.update(params)
    plt.plot(z, y, color=color)
    plt.xlabel("z")
    plt.ylabel("y")
    plt.title("Polynomial Plot")
    plt.show()


# Function Call
coeffs = np.array([0, -10, 1, -1, 1 / 1000])
z_range = [-10, 10]
color = "b"
# plot_polynomial(coeffs, z_range, color)


# *** Question 2 **
def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    print("Starting Exercise 2")
    random_state = np.random.RandomState(seed)
    x_min, x_max = z_range
    X = random_state.uniform(x_min, x_max, sample_size)

    epsilon = random_state.normal(0, sigma, sample_size)

    y = (
        coeffs[0]
        + coeffs[1] * X
        + coeffs[2] * X**2
        + coeffs[3] * X**3
        + coeffs[4] * X**4
        + epsilon
    )
    return X, y


# *** Question 4 **
def visualize_data(X, y, coeffs, z_range, title=""):
    print("Starting Exercise 4")
    plt.rcParams.update(params)
    plt.plot(X, y, "ro", alpha=0.7)

    z_min, z_max = z_range
    z = np.linspace(z_min, z_max, 100)
    w0, w1, w2, w3, w4 = coeffs
    y = w0 + w1 * z + w2 * z**2 + w3 * z**3 + w4 * z**4
    plt.plot(z, y, color="b")

    plt.xlabel("z")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


# class MyDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __getitem__(self, index):
#         return (self.X[index], self.y[index])

#     def __len__(self):
#         return len(self.X)

# class MyModel(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
#         super(MyModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         return x


if __name__ == "__main__":
    # *** Question 3 **
    print("Starting Exercise 3")
    coeffs = np.array([0, -10, 1, -1, 1 / 1000])
    z_range = [-3, 3]
    sigma = 0.5
    sample_size_train = 500
    sample_size_eval = 500
    seed_train = 0
    seed_eval = 1

    X_train, y_train = create_dataset(
        coeffs, z_range, sample_size_train, sigma, seed_train
    )
    X_eval, y_eval = create_dataset(coeffs, z_range, sample_size_eval, sigma, seed_eval)

    visualize_data(X_train, y_train, coeffs, z_range, title="Training Data")
    visualize_data(X_eval, y_eval, coeffs, z_range, title="Evaluation Data")

    # *** Question 5 **
    print("Starting Exercise 5")

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Use the first available GPU (cuda:0)
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    # Print the selected device
    print("Using device:", DEVICE)

    # Check if GPU is available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert the data to tensors and move to GPU
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_eval = torch.tensor(X_eval, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_eval = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    degree = 2
    model = nn.Sequential(nn.Linear(degree, 1)).to(DEVICE)

    loss_function = nn.MSELoss()

    learning_rate = 0.1
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = 3000

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        y_hat = model(X_train.pow(torch.arange(degree, dtype=torch.float32)).to(DEVICE))
        loss = loss_function(y_hat, y_train)
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_hat_eval = model(X_eval.pow(torch.arange(degree, dtype=torch.float32)).to(DEVICE))
            loss_eval = loss_function(y_hat_eval, y_eval)
            if epoch % 10 == 0:
                print(
                    "Epoch:",
                    epoch,
                    "Loss:",
                    loss.item(),
                    "Loss eval:",
                    loss_eval.item(),
                )

    # Print the final weights and bias
    print("Final weights:", model[0].weight, "Final bias:\n", model[0].bias)    

    # *** Question 6 **

    # *** Question 7 **

    # *** Question 8 **

    # *** Question 9 **

    # *** Question 10 **
