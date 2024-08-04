import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.CNNArchitectures import CustomCNN


def training_loop(
    device: str,  # Device to train the model on ('cuda' or 'cpu')
    model: torch.nn.Module,  # Model to train
    epoch: int,  # Current epoch number
    train_loader: DataLoader,  # DataLoader for training data
    x_epoch: list[int],  # List to store epoch numbers
    loss_fn: torch.nn.Module,  # Loss function to compute loss
    optimizer: torch.optim.Optimizer,  # Optimizer for updating model parameters
    y_loss: dict[str, list[float]]  # Dictionary to store training loss values
) -> None:
    """
    Performs the training process for one epoch.

    Parameters:
    - device (str): Device to train the model on ('cuda' or 'cpu').
    - model (torch.nn.Module): The model to train.
    - epoch (int): Current epoch number.
    - train_loader (DataLoader): DataLoader for training data.
    - x_epoch (list[int]): List to store epoch numbers.
    - loss_fn (torch.nn.Module): Loss function to compute loss.
    - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    - y_loss (dict[str, list[float]]): Dictionary to store training loss values.
    """
    model.to(device)
    model.train()
    print(f"Epoch: {epoch + 1}")
    x_epoch.append(epoch)
    running_loss = 0.0
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        train_data, train_labels = batch
        train_data = train_data.to(device)
        train_labels = train_labels.to(device)
        predictions = model(train_data)
        loss = loss_fn(predictions, train_labels)
        running_loss += loss.item()
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 499:  # Print loss every 500 batches
            avg_loss_across_batches = running_loss / 500
            print(
                "Batch {0}, Loss: {1:.3f}".format(
                    batch_idx + 1, avg_loss_across_batches
                )
            )
            running_loss = 0.0

    # Record average loss for the epoch
    y_loss["train"].append(epoch_loss / len(train_loader))
    print()


def evaluation_loop(
    device: str,  # Device to evaluate the model on ('cuda' or 'cpu')
    model: torch.nn.Module,  # Model to evaluate
    val_loader: DataLoader,  # DataLoader for validation data
    loss_fn: torch.nn.Module,  # Loss function to compute loss
    y_loss: dict[str, list[float]],  # Dictionary to store validation loss values
    accuracy_vals: list[float]  # List to store validation accuracy values
) -> None:
    """
    Evaluates the model on the validation set.

    Parameters:
    - device (str): Device to evaluate the model on ('cuda' or 'cpu').
    - model (torch.nn.Module): The model to evaluate.
    - val_loader (DataLoader): DataLoader for validation data.
    - loss_fn (torch.nn.Module): Loss function to compute loss.
    - y_loss (dict[str, list[float]]): Dictionary to store validation loss values.
    - accuracy_vals (list[float]): List to store validation accuracy values.
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, batch in enumerate(val_loader):
        test_data, test_labels = batch[0].to(device), batch[1].to(device)

        with torch.inference_mode():
            test_predictions = model(test_data)
            test_loss = loss_fn(test_predictions, test_labels)
            running_loss += test_loss.item()

            # Get the predicted class
            _, predicted = torch.max(test_predictions, 1)
            total_correct += (
                (predicted == test_labels).sum().item()
            )  # Compare predicted and actual labels
            total_samples += test_labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss_across_batches = running_loss / len(val_loader)
    y_loss["val"].append(avg_loss_across_batches)
    accuracy_vals.append(accuracy)

    print("Val Loss: {0:.3f}".format(avg_loss_across_batches))
    print(f"Total Correct: {total_correct}")
    print(f"Total Samples: {total_samples}")
    print("Accuracy: {0:.3f}".format(accuracy))
    print("***************************************************")
    print("\n")


def main() -> None:
    """
    Main function to run the model training and evaluation pipeline.
    """
    np.seterr(divide="ignore")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = 30
    torch.manual_seed(42)

    print("Loading in data...")
    # Load preprocessed datasets
    train_dataset = torch.load("./data/EMNIST/processed/train_dataset.pth")
    val_dataset = torch.load("./data/EMNIST/processed/val_dataset.pth")

    print("Creating data loaders...")
    # Create DataLoaders for training and validation sets
    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print("Initializing model...")
    # Initialize the model and print its summary
    model = CustomCNN()
    print(summary(model.to(device), input_size=(1, 28, 28)))

    print("Setting hyperparameters...")
    # Set the loss function, optimizer, and learning rate scheduler
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    print("Training model...")
    # Lists to track loss and accuracy values
    y_loss = {}
    y_loss["train"] = []
    y_loss["val"] = []
    x_epoch = []
    accuracy_vals = []
    for epoch in range(epochs):
        training_loop(
            device, model, epoch, train_loader, x_epoch, loss_fn, optimizer, y_loss
        )
        evaluation_loop(device, model, val_loader, loss_fn, y_loss, accuracy_vals)
        scheduler.step()

    print("Visualizing model metrics...")
    now = datetime.now()
    formatted_datetime = now.strftime("%Y_%B%d_%H_%M")

    # Plot and save training and validation loss metrics
    plt.figure()
    plt.plot(x_epoch, y_loss["train"], label="Train Loss")
    plt.plot(x_epoch, y_loss["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        f"./metrics/customcnn_{formatted_datetime}_loss_metrics.png",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()

    # Plot and save validation accuracy metrics
    plt.figure()
    plt.plot(x_epoch, accuracy_vals, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(
        f"./metrics/customcnn_{formatted_datetime}_acc_metrics.png",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()

    print("Saving trained model...")
    # Save the trained model
    os.makedirs("./models/", exist_ok=True)
    torch.save(model.state_dict(), f"./models/customcnn_{formatted_datetime}.pt")


if __name__ == "__main__":
    print("Running model training script...")
    main()
    print("Finished running model training script!")
