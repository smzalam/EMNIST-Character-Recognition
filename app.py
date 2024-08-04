import os
import io
import random
import streamlit as st
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from src.utils.CNNArchitectures import CustomCNN, ResNet34

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cnn_model = CustomCNN()
resnet_model = ResNet34(62)

cnn_model.load_state_dict(torch.load('./models/july31instance.pt', map_location=device))
resnet_model.load_state_dict(torch.load('./models/rnaug2instance.pt', map_location=device))

cnn_model.to(device)
resnet_model.to(device)
cnn_model.eval()
resnet_model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

combined_classes = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i",
    "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", "N/A",
]

# Function to make predictions using a model
def predict(image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to plot and display the processed image
def plot_processed_image_streamlit(image):
    image = transform(image).unsqueeze(0).cpu().numpy().squeeze()
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title("Processed Image for Model Input")
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, caption="Processed Image for Model Input")
    plt.close(fig)

# Function to convert PIL Image to bytes
def pil_image_to_bytes(img):
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        byte_im = buf.getvalue()
        return byte_im
    else:
        raise TypeError("Input must be a PIL Image")

# Function to get a random image for a specific label
def get_image_for_label(selected_label, df, test_dataset):
    indices = df[df['label'] == selected_label]['index'].tolist()
    
    if indices:
        random_index = random.choice(indices)
        img, label = test_dataset[random_index]
        
        img = transforms.ToPILImage()(img)
        return img, combined_classes[label]
    else:
        return None, None

# Function for validation step in model evaluation
def validation_step(batch, model, device):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)
    loss = F.cross_entropy(out, labels)
    _, preds = torch.max(out, dim=1)
    return {'val_loss': loss, 'preds': preds, 'labels': labels}

# Function to compute metrics at the end of an epoch
def validation_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()

    all_preds = torch.cat([x['preds'] for x in outputs])
    all_labels = torch.cat([x['labels'] for x in outputs])
    
    correct = torch.sum(all_preds == all_labels).item()
    total_samples = len(all_labels)
    epoch_acc = correct / total_samples
    
    precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
    
    return {
        'val_loss': epoch_loss.item(),
        'val_acc': epoch_acc,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1
    }

# Function to evaluate a model on the test dataset
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    outputs = []
    progress_bar = st.progress(0)
    total_batches = len(test_loader)
    
    for i, batch in enumerate(test_loader):
        outputs.append(validation_step(batch, model, device))
        progress_bar.progress((i + 1) / total_batches)
        
    return validation_epoch_end(outputs)

# Load test dataset and index DataFrame
data_dir = './data/EMNIST/processed/'
test_dataset = torch.load(os.path.join(data_dir, 'test_dataset.pth'))
df = pd.read_csv('./data/EMNIST/processed/test_dataset_index_df.csv')

# Streamlit app
st.title("EMNIST Letter Prediction App")

# Section for uploading and classifying images
# Section for uploading and classifying images
st.write("### Upload and Classification")
st.warning("**Note:** The models are primarily trained on the EMNIST dataset, which includes specific types of handwritten characters. Therefore, the accuracy of predictions may decrease when classifying images not from this dataset due to variability in writing styles, backgrounds, and image quality.")
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    st.write("### Uploaded Image")
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Classifying...")
    cnn_prediction_uploaded = predict(image, cnn_model)
    resnet_prediction_uploaded = predict(image, resnet_model)
    
    st.write(f"**CNN Model Prediction for uploaded image:** {combined_classes[cnn_prediction_uploaded]}")
    st.write(f"**ResNet Model Prediction for uploaded image:** {combined_classes[resnet_prediction_uploaded]}")

    plot_processed_image_streamlit(image)

st.write("### Select Class Example")
# Section for selecting a letter and displaying example images
selected_letter = st.selectbox("Select a letter:", combined_classes)

if selected_letter:
    st.write(f"Displaying example image for letter '{selected_letter}'")
    
    image, label = get_image_for_label(selected_letter, df, test_dataset)
    if image is not None:
        example_image_bytes = pil_image_to_bytes(image)
        st.image(example_image_bytes, caption=f"Example Image of letter '{selected_letter}'", use_column_width=True)

        st.write("Model predictions for example image:")
        cnn_prediction = predict(image, cnn_model)
        resnet_prediction = predict(image, resnet_model)
        
        st.write(f"**CNN Model Prediction for example image:** {combined_classes[cnn_prediction]}")
        st.write(f"**ResNet Model Prediction for example image:** {combined_classes[resnet_prediction]}")
    else:
        st.write(f"No example image found for the selected letter '{selected_letter}'")

st.write("### Evaluate Models on Test Dataset")

# Section for evaluating models on the test dataset
if st.button("Evaluate Models"):
    st.write("Evaluating models on test dataset...")
    st.write("Progress: ")
    batch_size = 128
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    st.write('Starting CNN evaluation...')
    cnn_metrics = evaluate(cnn_model, test_loader, device)
    st.write('Starting ResNet evaluation...')
    resnet_metrics = evaluate(resnet_model, test_loader, device)

    st.write("### CNN Model Metrics")
    st.write(cnn_metrics)
    
    st.write("### ResNet Model Metrics")
    st.write(resnet_metrics)
    
    metrics_names = ['val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1']
    cnn_metrics_list = [cnn_metrics[metric] for metric in metrics_names]
    resnet_metrics_list = [resnet_metrics[metric] for metric in metrics_names]

    st.write("### Model Performance Comparison")
    fig, ax = plt.subplots()
    x = np.arange(len(metrics_names))

    ax.bar(x - 0.2, cnn_metrics_list, 0.4, label='CNN')
    ax.bar(x + 0.2, resnet_metrics_list, 0.4, label='ResNet')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of CNN and ResNet Models on Test Dataset')
    ax.legend()

    st.pyplot(fig)
