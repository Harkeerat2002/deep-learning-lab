import platform
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from math import ceil

# Plot in Latex style
import matplotlib.pyplot as plt

params = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
}
plt.rcParams.update(params)

# Some lines are commented, uncomment them to use Neptune
# (N) import neptune

# (N) your credentials
# run = neptune.init_run(
#     project= # Projectname
#     api_token= # Token
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if platform.system() == "Darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("Wroking on ", device)

# Import MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="..", train=True, transform=transforms.ToTensor(), download=False
)           

test_dataset = torchvision.datasets.MNIST(
    root="..", train=False, transform=transforms.ToTensor(), download=False
)

batch_size = 128  # Hyperparameter!
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# Fully connected neural network with two hidden layers
class Model(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(Model, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # No need of softmax!
        return x


"""
Model
"""
input_size = 784  # 28 x 28 "rectified" images
hidden_size_1 = 256  # Hyperparameter!
hidden_size_2 = 128  # Hyperparameter!
num_classes = 10  # 10 digits = fixed

model = Model(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)

"""
Loss and optimizer
"""
learning_rate = 0.1  # Hyperparameter!
# (N) Save parameters to Neptune
# (N) params = {"learning_rate": learning_rate, "optimizer": "SGD"}
# (N) run["parameters"] = params

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
Training loop
"""
num_epochs = 2  # Hyperparameter!
softmax = nn.Softmax(dim=1)  # Need for accuracy computation
i_train = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        # Original shape --> [batch_size, 1, 28, 28] ---> [batch_size, 784]
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss_train = loss_fn(outputs, labels)
        # #(N) Load to neptune
        # (N) run["train/loss"].append(value=loss_train.item(), step=i_train)
        # Compute train accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy_train = correct / batch_size
        # #(N) Load to neptune
        # (N) run["train/accuracy"].append(value=accuracy_train, step=i_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            total_samples = len(test_dataset)
            # Here we do the evalutaion every 100 steps
            correct_test = 0
            n_batches = ceil(total_samples / batch_size)
            loss_test_total = 0
            with torch.no_grad():
                model.eval()
                for (
                    images,
                    labels,
                ) in test_loader:  # Batches, you cannot store evetything in the GPU
                    images = images.reshape(-1, 28 * 28).to(device)
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(images)
                    loss_test_total += loss_fn(outputs, labels).item()
                    # Compute train accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    correct_test += (predicted == labels).sum().item()
                    # # (N)  Load to neptune
                    # (N) run["test/loss"].append(value=loss_test_total / (len(test_dataset) /  batch_size), step=i_train)
                    # (N) run["test/accuracy"].append(value=accuracy_test_total / (len(test_dataset) / batch_size), step=i_train)
                    # i_train += 1
                    # Every 100 steps, print the train and test loss
                accuracy_test_total = correct_test / total_samples
                loss_test_total = loss_test_total / n_batches
                print(
                    "Epoch [{}/{}], Step [{}/{}]:".format(
                        epoch + 1, num_epochs, i + 1, len(train_loader)
                    )
                )
                print(
                    "\t Train Loss: {}; Accuracy: {} %".format(
                        round(loss_train.item(), 3), round(100 * accuracy_train)
                    )
                )
                print(
                    "\t Test Loss: {}; Accuracy: {} %".format(
                        round(loss_test_total, 3), round(100 * accuracy_test_total)
                    )
                )

    # Print accuracy at every epoch
    print(
        "Accuracy on the test set in epoch {}: {} %".format(
            epoch, round(100 * accuracy_test_total)
        )
    )

# #(N) Stop neptune
# (N) run.stop()

"""
Prediction of a chicken image
"""
from PIL import Image, ImageOps

img = Image.open("chicken_28x28.png")
# Tranform this image to a tensor
img = transforms.ToTensor()(img)
# Vectorize the image
img = img.reshape(-1, 28 * 28).to(device)
# Do the prediction
prediction = model(img)
# Get the softmax of such values
prediction = softmax(prediction)

plt.bar([i for i in range(10)], list(prediction.to("cpu").detach().numpy()[0]))
plt.savefig("prediction.png")
