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
    plt.plot(z, y, color=color)
    plt.xlabel("z")
    plt.ylabel("y")
    plt.show()


# Function Call
coeffs = np.array([0, -10, 1, -1, 1 / 1000])
z_range = [-10, 10]
color = "b"
# plot_polynomial(coeffs, z_range, color)


# *** Question 2 **
def create_dataset(
    coeffs,  # = w
    z_range,
    sample_size,
    sigma,
    seed,
):
    print("Starting Exercise 2")

    random_state = np.random.RandomState(seed)

    x_min, x_max = z_range

    X = random_state.uniform(x_min, x_max, (sample_size))

    X_final = np.array((np.ones(sample_size), X, X**2, X**3, X**4)).T

    epsilon = random_state.normal(0.0, sigma, sample_size)

    y = np.dot(X_final, coeffs) + epsilon

    return X_final, y


# *** Question 4 **
def visualize_data(X, y, coeffs, z_range, sample_size, title=""):
    print("Starting Exercise 4")
    plt.rcParams.update(params)
    plt.scatter(X[:, 1], y, color="r", label="Data")
    

    z_min, z_max = z_range
    z = np.linspace(z_min, z_max, sample_size)

    w0, w1, w2, w3, w4 = coeffs
    Y = w0 + w1 * z + w2 * z**2 + w3 * z**3 + w4 * z**4
    plt.plot(z, Y, color="b", label="True Function")

    plt.xlabel("z")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


class PolynomialRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(PolynomialRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim).to(device)

        self.device = device

    def forward(self, x):
        x = self.fc1(x)

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

    # visualize_data(
    #     X_train, y_train, coeffs, z_range, sample_size_train, title="Training Data"
    # )
    # visualize_data(
    #     X_eval, y_eval, coeffs, z_range, sample_size_eval, title="Evaluation Data"
    # )

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

    input_dim = 5

    output_dim = 1

    model = PolynomialRegressionModel(input_dim, output_dim, DEVICE)

    criteria = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Shape
    X_train = X_train.reshape(sample_size_train, input_dim)
    y_train = y_train.reshape(sample_size_train, output_dim)
    X_eval = X_eval.reshape(sample_size_eval, input_dim)
    y_eval = y_eval.reshape(sample_size_eval, output_dim)

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

    print(len(X_train), len(y_train))

    print(
        criteria(initial_model_value, torch.tensor(y_train).reshape(-1, 1).to(DEVICE))
    )

    train_losses = []
    eval_losses = []
    learned_coef = []

    # Train the model
    num_epochs = 5000

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criteria(y_pred, y_train)
        loss.backward()
        optimizer.step()
        model.eval()

        with torch.no_grad():
            y_pred_eval = model(X_eval)
            eval_loss = criteria(y_pred_eval, y_eval)
            eval_losses.append(eval_loss.item())
            train_losses.append(loss.item())

            weights = []
            for name, param in model.named_parameters():
                if param.requires_grad and "weight" in name:
                    weights.append(param.data.cpu().numpy()[0])
                    break

            learned_coef.append(weights)

            if epoch % 1000 == 0:
                print("Epoch:", epoch, "Loss:", loss.item())

            # if loss.item() < 0.5:
            #    break

    print("Final loss:", loss.item())

    # *** Question 6 **
    # plt.rcParams.update(params)
    # plt.plot(train_losses, label="Training Loss")
    # plt.plot(eval_losses, label="Evaluation Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

    # *** Question 7 **
    estimated_weights = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Weight name:", name, "Weight value:", param.data.cpu().numpy()[0])
            estimated_weights = (param.data.cpu().numpy()[0])
            break
    # print("Printing Estimated Weights", estimated_weights)
    
    # estimated_weights = np.array(estimated_weights) /
    
    # z_min, z_max = z_range
    # z = np.linspace(z_min, z_max, sample_size_eval)

    # w0, w1, w2, w3, w4 = coeffs
    # W0, W1, W2, W3, W4 = estimated_weights
    
    # y = w0 + w1 * z + w2 * z**2 + w3 * z**3 + w4 * z**4
    # Y = W0 + W1 * z + W2 * z**2 + W3 * z**3 + W4 * z**4
    
    # plt.plot(z, y, color="b", label="True Coefficients")
    # plt.plot(z, Y, color="r", label="Estimated Coefficients")
    # estimated_y = model(X_eval)
    
    
    # plt.xlabel("z")
    # plt.ylabel("y")
    # plt.title("Estimated Coefficients vs True Coefficients")
    # plt.legend()
    # plt.show()
    
    

    # *** Question 8 **
    print(len(learned_coef))
    print(len(learned_coef[0][0]))

    learned_w0 = []
    learned_w1 = []
    learned_w2 = []
    learned_w3 = []
    learned_w4 = []
    for i in range(len(learned_coef)):
        learned_w0.append(learned_coef[i][0][0])
        learned_w1.append(learned_coef[i][0][1])
        learned_w2.append(learned_coef[i][0][2])
        learned_w3.append(learned_coef[i][0][3])
        learned_w4.append(learned_coef[i][0][4])

    # Extract true w0 value
    true_w0 = coeffs[0]
    true_w1 = coeffs[1]
    true_w2 = coeffs[2]
    true_w3 = coeffs[3]
    true_w4 = coeffs[4]

    # Create a list of epoch numbers for x-axis
    epochs = list(range(len(learned_w0)))

    # Plot the estimated and true w0 values
    plt.plot(epochs, learned_w0, label="Estimated w0")
    plt.plot(epochs, [true_w0] * len(epochs), label="True w0")
    plt.xlabel("Epoch")
    plt.ylabel("w0")
    plt.legend()
    plt.show()
    
    plt.plot(epochs, learned_w1, label="Estimated w1")
    plt.plot(epochs, [true_w1] * len(epochs), label="True w1")
    plt.xlabel("Epoch")
    plt.ylabel("w1")
    plt.legend()
    plt.show()
    
    plt.plot(epochs, learned_w2, label="Estimated w2")
    plt.plot(epochs, [true_w2] * len(epochs), label="True w2")
    plt.xlabel("Epoch")
    plt.ylabel("w2")
    plt.legend()
    plt.show()
    
    plt.plot(epochs, learned_w3, label="Estimated w3")
    plt.plot(epochs, [true_w3] * len(epochs), label="True w3")
    plt.xlabel("Epoch")
    plt.ylabel("w3")
    plt.legend()
    plt.show()
    
    plt.plot(epochs, learned_w4, label="Estimated w4")
    plt.plot(epochs, [true_w4] * len(epochs), label="True w4")
    plt.xlabel("Epoch")
    plt.ylabel("w4")
    plt.legend()
    plt.show()


    # *** Question 9 **

    # *** Question 10 **
