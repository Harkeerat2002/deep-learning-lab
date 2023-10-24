import numpy as np


def create_dataset(
    sample_size=10, sigma=0.1, w_star=1, b_star=1, x_range=(-1, 1), seed=0
):
    # Set the random state in numpy
    random_state = np.random.RandomState(seed)
    # Unpack the values in x_range
    x_min, x_max = x_range
    # Sample sample_size points from a uniform distribution
    X = random_state.uniform(x_min, x_max, (sample_size))
    # Compute hat(y)
    y_hat = X * w_star + b_star
    # Compute y (Add Gaussian noise)
    y = y_hat + random_state.normal(0.0, sigma, (sample_size))
    return X, y


# Generate the training data
num_samples_train = 200
sigma = 4
w_star = 3
b_star = 2
seed_train = 0
X_train, y_train = create_dataset(
    sample_size=num_samples_train,
    sigma=0.1,
    w_star=w_star,
    b_star=b_star,
    x_range=(-1, 1),
    seed=seed_train,
)

# Generate the validation data form the same distribution but with a different seed
num_samples_validation = 200
seed_validation = 42
X_val, y_val = create_dataset(
    sample_size=num_samples_validation,
    sigma=0.1,
    w_star=w_star,
    b_star=b_star,
    x_range=(-1, 1),
    seed=seed_validation,
)

import matplotlib.pyplot as plt

plt.figure()  # Open a new figure
plt.plot(X_train, y_train, "ro", alpha=0.7)  # Scatter plot of X_train and y_train
plt.plot(X_val, y_val, "bo", alpha=0.7)  # Scatter plot of X_val and y_val
plt.savefig("my_figure.png")
# Or plt.savefig("my_figure.png") to save the figure in a file

import platform
import torch

# If you are using a M1/M2 MacBook..
if torch.cuda.is_available():
    # Use the first available GPU (cuda:0)
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

# Check which devide you have active with a print
print("Device:")
print(DEVICE)

# Create the model
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)  # Dimension of input: 1, dimension of output: 1
loss_fn = nn.MSELoss()  # Set MSE as loss function
learning_rate = 0.1
# Optimize all the parameters of the model
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print("Initial w:", model.weight, "Initial b:\n", model.bias)
print("Value in x = 1:", model(torch.tensor([1.0])))  # This computes w * 1 + b

# Loss evaluation
initial_model_value = model(
    torch.tensor(X_train, dtype=torch.float32).reshape(num_samples_train, 1)
)
print(loss_fn(initial_model_value, torch.tensor(y_train).reshape(-1, 1)))

# Prepare for the training loop
# Shape expected by nn.Linear
X_train = X_train.reshape(num_samples_train, 1)
y_train = y_train.reshape(num_samples_train, 1)
X_eval = X_val.reshape(num_samples_validation, 1)
y_eval = y_val.reshape(num_samples_validation, 1)

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

model = model.to(DEVICE)  # Move the model to the device you want to use

n_steps = 50
import time

start = time.time()
for step in range(n_steps):
    # *** Training ***
    model.train()  # Set the model in training mode
    # Set the gradient to 0
    optimizer.zero_grad()  # Or model.zero_grad()
    # Compute the output of the model
    y_hat = model(X_train)
    # Compute the loss
    loss = loss_fn(y_hat, y_train)
    # Compute the gradient
    loss.backward()
    # Update the parameters
    optimizer.step()
    # *** Evaluation ***
    # Here we do things that do not contribute to the gradient computation
    model.eval()  # Set the model in evaluation mode
    with torch.no_grad():
        # Compute the output of the model
        y_hat_eval = model(X_eval)
        # Compute the loss
        loss_eval = loss_fn(y_hat_eval, y_eval)
        # Compute the output of the model
        # Every 10 steps, print the loss
        if step % 10 == 0:
            print("Step:", step, "- Loss eval:", loss_eval.item())
print(
    "Training done, with an evaluation loss of {} in time {}".format(
        loss_eval.item(), time.time() - start
    )
)
# Get the final value of the parameters
print("Final w:", model.weight, "Final b:\n", model.bias)

# Linear regression with sklearn
from sklearn.linear_model import LinearRegression

start = time.time()
reg = LinearRegression().fit(X_train.to("cpu").numpy(), y_train.to("cpu").numpy())
print("Training done in {}".format(time.time() - start))
print("Final w:", reg.coef_, "Final b:\n", reg.intercept_)
