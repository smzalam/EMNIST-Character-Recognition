import numpy as np
import torch
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset

# Function to gather training and testing datasets from various EMNIST splits
def gather_data() -> list:
    """
    Collects training and testing datasets from various EMNIST splits and applies transformations.

    Returns:
    - list: A list containing two elements:
      - Training datasets (list of EMNIST datasets)
      - Testing datasets (list of EMNIST datasets)
    """
    # Define the transformations: convert to tensor and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    print("Gathering training data...")

    # Load training datasets from different EMNIST splits
    train_datasets_tensors = [
        EMNIST(
            root="./data/",
            split="letters",
            train=True,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="byclass",
            train=True,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="bymerge",
            train=True,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="balanced",
            train=True,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="digits",
            train=True,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="mnist",
            train=True,
            download=False,
            transform=transform,
        ),
    ]

    print("Gathering testing data...")

    # Load testing datasets from different EMNIST splits
    test_datasets_tensors = [
        EMNIST(
            root="./data/",
            split="letters",
            train=False,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="byclass",
            train=False,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="bymerge",
            train=False,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="balanced",
            train=False,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="digits",
            train=False,
            download=False,
            transform=transform,
        ),
        EMNIST(
            root="./data/",
            split="mnist",
            train=False,
            download=False,
            transform=transform,
        ),
    ]

    print("Finished gathering data...")

    return [train_datasets_tensors, test_datasets_tensors]

# Function to standardize label ranges across different splits
def standardizing_label_ranges(datasets: list | Dataset) -> list | Dataset:
    """
    Standardizes label ranges across different EMNIST splits to ensure consistency.

    Parameters:
    - datasets (list or Dataset): The datasets to be standardized.

    Returns:
    - list or Dataset: The datasets with standardized labels.
    """
    # Define the combined classes in a standard order
    combined_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i",
        "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
        "y", "z", "N/A",
    ]

    # Create a mapping from original class indices to new class indices
    def create_label_mapping(original_classes, combined_classes: list):
        return {original_class: combined_classes.index(original_class) for original_class in original_classes}

    # Function to remap labels in a dataset
    def remap_labels(dataset, label_mapping):
        new_targets = [label_mapping[dataset.classes[label]] for label in dataset.targets]
        dataset.targets = torch.tensor(new_targets)

    # Remap labels for each dataset in the list
    for dataset in datasets:
        label_mapping = create_label_mapping(dataset.classes, combined_classes)
        remap_labels(dataset, label_mapping)

    return datasets

# Function to combine multiple datasets into one
def combining_datasets(datasets: list) -> ConcatDataset:
    """
    Combines multiple datasets into one using PyTorch's ConcatDataset.

    Parameters:
    - datasets (list): List of datasets to be combined.

    Returns:
    - ConcatDataset: A combined dataset containing all the input datasets.
    """
    combined_dataset = ConcatDataset([dataset for dataset in datasets])
    return combined_dataset

# Main function to execute the data processing pipeline
def main():
    """
    Executes the data processing pipeline including loading, standardizing, combining, and saving datasets.
    """
    np.seterr(divide="ignore")

    print("Loading in data...")
    train_dataset, test_dataset = gather_data()

    print("Performing label standardization...")
    train_dataset = standardizing_label_ranges(train_dataset)
    test_dataset = standardizing_label_ranges(test_dataset)

    print("Combining datasets...")
    combined_train_dataset = combining_datasets(train_dataset)
    combined_test_dataset = combining_datasets(test_dataset)

    print("Saving combined dataset...")
    torch.save(combined_train_dataset, './data/EMNIST/processed/combined_train_dataset.pth')
    torch.save(combined_test_dataset, './data/EMNIST/processed/combined_test_dataset.pth')

# Entry point of the script
if __name__ == "__main__":
    print("Running data preprocessing script...")
    main()
    print("Finished running data preprocessing script!")
