"""
Assignment 2
Student: NAME SURNAME
"""
# *** Packges ***
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
from art import text2art
import os


# *** Functions ***
def imshow(img, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(name)
    plt.savefig(dir + name + ".png")


if __name__ == "__main__":
    print(text2art("DLL: Assignment 2"))

    # Write your code here
    if torch.cuda.is_available():
        print("GPU is available, and device is:", torch.cuda.get_device_name(0))
    else:
        print("GPU is not available, CPU is used")

    """
    DON'T MODIFY THE SEED
    """
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)

    print("-------------------------")
    print("[bold cyan]QUESTION 1.1: DATA [/bold cyan]")
    print("-------------------------")

    print("[bold green]Question 1.1.1 (5pts) [/bold green]")

    # Loading the Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Observing the data
    first_images = {}

    for images, labels in trainset:
        class_name = classes[labels]
        if class_name not in first_images:
            first_images[class_name] = images

        if len(first_images) == len(classes):
            break

    dir1 = "./figures/1.1/"
    for class_name, image in first_images.items():
        imshow(image, dir1, class_name)
    print(
        "First images of each class are saved in the [bold yellow]figures/1.1[bold yellow] folder"
    )

    # Creating Histogram to show the distribution of the data

    ## Initializing the counts
    train_counts = {class_name: 0 for class_name in classes}
    test_counts = {class_name: 0 for class_name in classes}

    for images, labels in trainset:
        class_name = classes[labels]
        train_counts[class_name] += 1

    for images, labels in testset:
        class_name = classes[labels]
        test_counts[class_name] += 1

    ## Plotting the histogram
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(train_counts.keys(), train_counts.values(), color="g")
    plt.title("Train Dataset")
    plt.xlabel("Classes")
    plt.ylabel("Counts")

    plt.subplot(1, 2, 2)
    plt.bar(test_counts.keys(), test_counts.values(), color="b")
    plt.title("Test Dataset")
    plt.xlabel("Classes")
    plt.ylabel("Counts")

    plt.tight_layout()

    if not os.path.exists(dir1):
        os.makedirs(dir1)
    plt.savefig(dir1 + "histogram.png")

    print(
        "Histogram of the data is saved in the [bold yellow]figures/1.1[bold yellow] folder"
    )

    print("\n[bold green]Question 1.1.2 (5pts) [/bold green]")

    # Creating the dataloader to convert the data to tensor

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Getting a sample from the dataset to check if the data is correct type
    data_sample, label_sample = next(iter(trainloader))

    print("Data type of the sample is:", data_sample.dtype)

    if not isinstance(data_sample, torch.Tensor):
        data_sample = torch.tensor(data_sample)
        print("Data type of the sample is changed to:", data_sample.dtype)

    if not isinstance(label_sample, torch.Tensor):
        label_sample = torch.tensor(label_sample)
        print("Data type of the sample is changed to:", label_sample.dtype)

    # Print Dimension of the tensor
    print("Dimension of the data tensor is: ", data_sample.shape)

    print("\n[bold green]Question 1.1.3 (5pts) [/bold green]")
    transformNorm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])
    
    trainset.transform = transformNorm
    testset.transform = transformNorm
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    pixels_red = []
    pixels_green = []
    pixels_blue = []

    
    for data, _ in trainset:
        red, green, blue = data.split(1)
        pixels_red.extend(red.flatten().tolist())
        pixels_green.extend(green.flatten().tolist())
        pixels_blue.extend(blue.flatten().tolist())

    # Calculate the mean and standard deviation for each channel
    mean_red = sum(pixels_red) / len(pixels_red)
    std_red = (sum((i - mean_red) ** 2 for i in pixels_red) / len(pixels_red)) ** 0.5
    mean_green = sum(pixels_green) / len(pixels_green)
    std_green = (sum((i - mean_green) ** 2 for i in pixels_green) / len(pixels_green)) ** 0.5
    mean_blue = sum(pixels_blue) / len(pixels_blue)
    std_blue = (sum((i - mean_blue) ** 2 for i in pixels_blue) / len(pixels_blue)) ** 0.5
    
    console = Console()
    
    # Create a table
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns to the table
    table.add_row("", "Before Normalization", "")
    table.add_column("Channel", style="cyan")
    table.add_column("Mean", justify="center", style="green")
    table.add_column("Std Deviation", justify="center", style="blue")

    # Add data to the table
    table.add_row("Red", f"{mean_red:.4f}", f"{std_red:.4f}")
    table.add_row("Green", f"{mean_green:.4f}", f"{std_green:.4f}")
    table.add_row("Blue", f"{mean_blue:.4f}", f"{std_blue:.4f}")
    
    # Print the table
    console.print(table)

    # Create a normalization transform
    normalize = transforms.Normalize((mean_red, mean_green, mean_blue), (std_red, std_green, std_blue))

    # Combine the existing transformations with the new normalization transform
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Apply the composed transform to the training and test datasets
    trainset.transform = transform
    testset.transform = transform
    
    # Calculating the mean and std again
    # Initialize lists to store pixel values for each channel
    pixels_red = []
    pixels_green = []
    pixels_blue = []

    # Iterate over the entire training set
    for data, _ in trainset:
        red, green, blue = data.split(1)
        pixels_red.extend(red.flatten().tolist())
        pixels_green.extend(green.flatten().tolist())
        pixels_blue.extend(blue.flatten().tolist())

    # Convert lists to tensors
    pixels_red = torch.tensor(pixels_red)
    pixels_green = torch.tensor(pixels_green)
    pixels_blue = torch.tensor(pixels_blue)

    # Calculate the mean and standard deviation for each channel
    new_mean_red = torch.mean(pixels_red)
    new_std_red = torch.std(pixels_red)
    new_mean_green = torch.mean(pixels_green)
    new_std_green = torch.std(pixels_green)
    new_mean_blue = torch.mean(pixels_blue)
    new_std_blue = torch.std(pixels_blue)

    

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")

    # Add columns to the table
    table.add_column("Channel", style="cyan")
    table.add_column("Mean", justify="center", style="green")
    table.add_column("Std Deviation", justify="center", style="blue")

    # Add data to the table
    table.add_row("Red", f"{new_mean_red:.4f}", f"{new_std_red:.4f}")
    table.add_row("Green", f"{new_mean_green:.4f}", f"{new_std_green:.4f}")
    table.add_row("Blue", f"{new_mean_blue:.4f}", f"{new_std_blue:.4f}")
    
    # Print the table
    console.print(table)

    """
    Code for bonus question
    """
    for seed in range(10):
        torch.manual_seed(seed)
        # Train the models here
