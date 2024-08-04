from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.CNNArchitectures import CustomCNN
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def validation_step(device, model, batch):
    """
    Performs a single validation step on a batch of data.
    
    Args:
        device (torch.device): The device to perform computation on.
        model (torch.nn.Module): The model to evaluate.
        batch (tuple): A tuple containing images and labels.
    
    Returns:
        dict: Contains validation loss, predictions, and true labels.
    """
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    _, preds = torch.max(out, dim=1)
    return {'val_loss': loss, 'preds': preds, 'labels': labels}

def validation_epoch_end(outputs):
    """
    Aggregates the results of validation steps to compute metrics for the entire epoch.
    
    Args:
        outputs (list of dicts): A list of dictionaries containing loss, predictions, and labels for each batch.
    
    Returns:
        dict: Contains average validation loss, accuracy, precision, recall, and F1 score, as well as per-class metrics.
    """
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses

    all_preds = torch.cat([x['preds'] for x in outputs])
    all_labels = torch.cat([x['labels'] for x in outputs])
    
    # Calculate accuracy
    correct = torch.sum(all_preds == all_labels).item()
    total_samples = len(all_labels)
    epoch_acc = correct / total_samples
    
    # Convert tensors to numpy arrays for metric calculations
    all_preds_cpu = all_preds.cpu().numpy()
    all_labels_cpu = all_labels.cpu().numpy()

    # Calculate precision, recall, and f1-score
    precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    
    class_precision = precision_score(all_labels_cpu, all_preds_cpu, average=None)
    class_recall = recall_score(all_labels_cpu, all_preds_cpu, average=None)
    class_f1 = f1_score(all_labels_cpu, all_preds_cpu, average=None)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels.cpu(), all_preds.cpu(), normalize='true')
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])], 
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'val_loss': epoch_loss.item(),
        'val_acc': epoch_acc,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
    }

@torch.no_grad()
def evaluate(device, model, val_loader):
    """
    Evaluates the model on the validation data.
    
    Args:
        device (torch.device): The device to perform computation on.
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation dataset.
    
    Returns:
        dict: Contains validation metrics.
    """
    model.eval()
    outputs = [validation_step(device, model, batch) for batch in val_loader]
    return validation_epoch_end(outputs)

def main(modelName: str):
    """
    Main function to load a trained model, evaluate it, and display metrics.
    
    Args:
        modelName (str): The filename of the model to evaluate.
    """
    np.seterr(divide="ignore")  # Ignore division errors in NumPy
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise use CPU
    batch_size = 128
    torch.manual_seed(42)  # Set random seed for reproducibility

    # Ensure model directory exists
    model_dir = './models/'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    # Load model
    model = CustomCNN()  # Initialize the CustomCNN model
    model_path = os.path.join(model_dir, modelName)  # Construct the full path to the model file
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model.to(device)  # Move model to the appropriate device (GPU/CPU)

    # Ensure data directory exists
    data_dir = './data/EMNIST/processed/'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Load test dataset
    test_dataset = torch.load(os.path.join(data_dir, 'test_dataset.pth'))  # Load the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)  # Create DataLoader for the test dataset

    # Evaluate model
    metrics = evaluate(device, model, test_loader)  # Perform evaluation and get metrics
    print(f"Validation Loss: {metrics['val_loss']:.4f}")  # Print validation loss
    print(f"Validation Accuracy: {metrics['val_acc']:.4f}")  # Print validation accuracy
    print(f"Validation Precision: {metrics['val_precision']:.4f}")  # Print validation precision
    print(f"Validation Recall: {metrics['val_recall']:.4f}")  # Print validation recall
    print(f"Validation F1 Score: {metrics['val_f1']:.4f}")  # Print validation F1 score

    # Creating a DataFrame for class metrics
    now = datetime.now()  # Get current datetime
    formatted_datetime = now.strftime("%Y_%B%d_%H_%M")  # Format datetime for filename
    class_names = [f'Class {i}' for i in range(len(metrics['class_precision']))]  # Generate class names
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': metrics['class_precision'],
        'Recall': metrics['class_recall'],
        'F1 Score': metrics['class_f1']
    })  # Create DataFrame for class metrics

    print(metrics_df)  # Print class metrics DataFrame

    # Plotting Precision, Recall, and F1 Score
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))  # Create a figure with 3 subplots

    # Precision
    ax[0].bar(range(len(metrics['class_precision'])), metrics['class_precision'], color='blue')  # Bar plot for precision
    ax[0].set_title('Precision per Class')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Precision')

    # Recall
    ax[1].bar(range(len(metrics['class_recall'])), metrics['class_recall'], color='green')  # Bar plot for recall
    ax[1].set_title('Recall per Class')
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Recall')

    # F1 Score
    ax[2].bar(range(len(metrics['class_f1'])), metrics['class_f1'], color='red')  # Bar plot for F1 score
    ax[2].set_title('F1 Score per Class')
    ax[2].set_xlabel('Class')
    ax[2].set_ylabel('F1 Score')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plots
    plt.savefig(
        f"./metrics/customcnn_{formatted_datetime}_class_metrics.png",
        bbox_inches="tight",
        pad_inches=0.3,
    )  # Save the figure with metrics plots


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate a trained model.')
    parser.add_argument('modelName', type=str, help='The filename of the model to evaluate.')
    
    # Parse arguments
    args = parser.parse_args()
    
    print("Running model evaluation script...")
    main(modelName=args.modelName)
    print("Finished running model evaluation script!")
