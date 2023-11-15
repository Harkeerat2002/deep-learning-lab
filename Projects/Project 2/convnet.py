'''
Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Disclaimer: not the best network for CIFAR10, but it is simple and it works.
Just for educational purposes.
'''
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch

# Set the seed in torch
torch.manual_seed(42)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2)
        fc_size = 2 * 2 * 3
        self.fc1 = nn.Linear(fc_size, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.fc2(F.relu(self.fc1(x)))
        return x

# Classify dogs and cats in CIFAR 10 and observe the meaning of each layer.
# Upload the CIFAR10 dataset.
transform = transforms.ToTensor()

# Download CIFAR, train dataset
# We are not interested in the test set now
batch_size = 16
trainset = torchvision.datasets.CIFAR10(root='./data2/', train=True,
                                        download=True, transform=transform)
# Select only dogs and cats
I_dog_cats_train = [i for i in range(len(trainset)) if trainset[i][1] in [3, 5]]

trainset = [trainset[i] for i in I_dog_cats_train]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)



# Define the network
net = ConvNet()

# Define the loss function and the optimizer
import torch.optim as optim
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
n_epochs = 20

# Training loop
for epoch in range(n_epochs):  # loop over the dataset multiple times
    for i, (images, labels) in enumerate(trainloader):
        labels = torch.tensor([0 if label == 3 else 1 for label in labels])
        net.train()
        # Forward pass
        outputs = net(images)
        loss_train = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        if (i + 1) % 300 == 0:
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy_train = correct / batch_size
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch + 1, n_epochs, i, len(trainloader), loss_train.item(), accuracy_train * 100))

print("Training finished!")

'''
On a single image, we can see the output of each layer.
'''
image, label = trainset[0]
trained_params = net.state_dict()
# **** Original image ****
# Tranform it to a PIL image
transform = transforms.ToPILImage()
# convert the tensor to PIL image using above transform
imgage_plot = transform(image)
# save the PIL image
imgage_plot.save('image_0.png')
# **** First layer ****
conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
conv1.load_state_dict({'weight' : trained_params['conv1.weight'], 'bias' : trained_params['conv1.bias']})
image_1 = conv1(image)
# convert the tensor to PIL image using above transform
imgage_1_plot = transform(image_1)
# save the PIL image
imgage_1_plot.save('image_1_plot.png')

# **** Second layer ****
conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
conv2.load_state_dict({'weight' : trained_params['conv2.weight'], 'bias' : trained_params['conv2.bias']})
image_2 = conv2(image_1)
# convert the tensor to PIL image using above transform
imgage_2_plot = transform(image_2)
# save the PIL image
imgage_2_plot.save('image_2_plot.png')

# **** First layer ****
conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2)
conv3.load_state_dict({'weight' : trained_params['conv3.weight'], 'bias' : trained_params['conv3.bias']})
image_3 = conv3(image_2)
# convert the tensor to PIL image using above transform
imgage_3_plot = transform(image_3)
# save the PIL image
imgage_3_plot.save('image_3_plot.png')