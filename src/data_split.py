import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, random_split

def split_data(dataset: ConcatDataset | Dataset) -> list:
    """
    Splits a given dataset into training and validation sets based on specified ratios.

    Parameters:
    - dataset (ConcatDataset or Dataset): The dataset to be split.

    Returns:
    - list: A list containing the training and validation datasets.
    """
    # Define the sizes of the splits
    total_size = len(dataset)
    split_ratios = [0.9, 0.1]  # Ratios for splitting (e.g., 90% training, 10% validation)
    split_sizes = [int(total_size * ratio) for ratio in split_ratios]

    # Adjust the last split size to match the total size exactly
    split_sizes[-1] = total_size - sum(split_sizes[:-1])

    # Split the dataset into training and validation datasets
    train_dataset, val_dataset = random_split(dataset, split_sizes)

    return [train_dataset, val_dataset]

def main():
    """
    Manages the loading, splitting, and saving of datasets.
    """
    # Ignore divide errors for the numpy operations
    np.seterr(divide="ignore")

    print("Loading in data...")
    # Load the combined training and testing datasets from disk
    combined_train_dataset = torch.load('./data/EMNIST/processed/combined_train_dataset.pth')
    combined_test_dataset = torch.load('./data/EMNIST/processed/combined_test_dataset.pth')

    print("Splitting data...")
    # Split the combined training dataset into training and validation datasets
    train_dataset, val_dataset = split_data(combined_train_dataset)
    test_dataset = combined_test_dataset
    
    print("Saving split datasets...")
    # Save the resulting datasets to disk
    torch.save(train_dataset, './data/EMNIST/processed/train_dataset.pth')
    torch.save(val_dataset, './data/EMNIST/processed/val_dataset.pth')
    torch.save(test_dataset, './data/EMNIST/processed/test_dataset.pth')

if __name__ == "__main__":
    print("Running data preprocessing script...")
    # Execute the main function
    main()
    print("Finished running data preprocessing script!")
