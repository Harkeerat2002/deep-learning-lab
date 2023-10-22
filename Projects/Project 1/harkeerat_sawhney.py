"""
Assignment 1: Polynomial Regression
Student: Harkeerat Singh Sawhney
"""
# Libraries:
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn as nn
import torch.nn.functional as F


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
def create_dataset(
    coeffs=[1, 2, 3, 4, 5],
    z_range=np.array([0, -10, 1, -1, 1 / 1000]),
    sample_size=10,
    sigma=0.1,
    seed=42,
):
    print("Starting Exercise 2")

    random_state = np.random.RandomState(seed)

    x_min, x_max = z_range

    X = random_state.uniform(x_min, x_max, (sample_size))

    epsilon = random_state.normal(0.0, sigma, sample_size)

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


class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(PolynomialRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100, bias=True).to(device)
        self.fc2 = nn.Linear(100, output_dim, bias=True).to(device)
        self.device = device

    def forward(self, x):
        x = torch.relu(self.fc1(x)).to(self.device)
        x = self.fc2(x).to(self.device)
        return x


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

    # visualize_data(X_train, y_train, coeffs, z_range, title="Training Data")
    # visualize_data(X_eval, y_eval, coeffs, z_range, title="Evaluation Data")

    # *** Question 5 **
    print("Starting Exercise 5")

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        # Use the first available GPU (cuda:0)
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    DEVICE = torch.device("cuda:0")

    # Print the selected device
    print("Using device:", DEVICE)

    # Reshaping the Data to Polynomial Features
    X_train = X_train.reshape(-1, 1)
    X_eval = X_eval.reshape(-1, 1)

    degree = 4

    # X_train = model.fit_transform(X_train)
    # X_eval = model.fit_transform(X_eval)
    input_dim = X_train.shape[1]
    output_dim = 1

    model = PolynomialRegressionModel(input_dim, output_dim, DEVICE)

    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Shape expected by nn.Linear
    X_train = X_train.reshape(sample_size_train, 1)
    y_train = y_train.reshape(sample_size_train, 1)
    X_eval = X_eval.reshape(sample_size_eval, 1)
    y_eval = y_eval.reshape(sample_size_eval, 1)

    # Convert everything to torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_eval = torch.tensor(X_eval, dtype=torch.float32)
    y_eval = torch.tensor(y_eval, dtype=torch.float32)

    # Move everything to the device you want to use
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_eval = X_eval.to(DEVICE)
    y_eval = y_eval.to(DEVICE)

    initial_model_value = model(X_train)

    print(
        criteria(initial_model_value, torch.tensor(y_train).reshape(-1, 1).to(DEVICE))
    )

    # Train the model
    num_epochs = 3000

    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criteria(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs_eval = model(X_eval)
            loss_eval = criteria(outputs_eval, y_eval)

            if (epoch + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, loss.item()
                    )
                )

                if loss.item() < 0.5:
                    break

    print("Final loss:", loss.item())

    # *** Question 6 **

    # *** Question 7 **

    # *** Question 8 **

    # *** Question 9 **

    # *** Question 10 **
