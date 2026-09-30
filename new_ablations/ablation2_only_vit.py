#needs model and loss func change -> only vit + standard cross entropy loss (non-weighted) -> done
#make change here of -> only vit + standard cross entropy loss (weighted) -> done; results have come


# Add these lines at the very top of your script (after imports)

import warnings
warnings.filterwarnings('ignore')

import os
from PIL import Image
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle

# ================================================================================
# REVISED FUNCTION FOR GRAD-CAM (FIXING ATTRIBUTEERROR)
# This version corrects the bug by properly handling the list of labels
# and accepting the class names as an argument.
# ================================================================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision


def generate_roc_pr_curves(model, test_loader, device, class_names):
    """Generates and saves ROC and Precision-Recall curve plots using scikit-learn."""
    print("Generating ROC and Precision-Recall curves...")
    model.eval()
    y_true = []
    y_probas = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            # Use softmax to get probabilities
            probas = F.softmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_probas.extend(probas)

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    n_classes = len(class_names)

    # --- ROC Curve Generation ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # Binarize the labels for the current class
        y_true_class = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig("roc_curves.png")
    plt.close()
    print("Saved ROC curves to roc_curves.png")

    # --- Precision-Recall Curve Generation ---
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        y_true_class = (y_true == i).astype(int)
        precision[i], recall[i], _ = precision_recall_curve(y_true_class, y_probas[:, i])
        average_precision[i] = average_precision_score(y_true_class, y_probas[:, i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0} (AP = {1:0.2f})'
                       ''.format(class_names[i], average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves per Class')
    plt.legend(loc="best")
    plt.savefig("pr_curves.png")
    plt.close()
    print("Saved Precision-Recall curves to pr_curves.png")


# ================================================================================
# REVISED FUNCTION FOR GRAD-CAM (NOW COMPATIBLE WITH ViT)
# This version accepts an optional reshape_transform for ViT compatibility.
# ================================================================================
def generate_grad_cam_visualizations(model,
                                     test_loader,
                                     device,
                                     target_layers,
                                     class_names,
                                     file_name="gradcam_heatmaps.png",
                                     reshape_transform=None):  # New optional argument
    """Generates and saves Grad-CAM visualizations for one sample from each class."""
    print(f"Generating Grad-CAM visualizations for {file_name}...")
    model.eval()

    num_classes = len(class_names)
    images_per_class = {i: None for i in range(num_classes)}

    for images, labels in test_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if images_per_class[label] is None:
                images_per_class[label] = images[i]
        if all(v is not None for v in images_per_class.values()):
            break

    # Pass the reshape_transform to the GradCAM constructor
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=reshape_transform)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Grad-CAM: Model Attention on Different Dementia Stages', fontsize=16)

    for i, (class_idx, image_tensor) in enumerate(images_per_class.items()):
        if image_tensor is None:
            print(f"Warning: Could not find an image for class {class_names[class_idx]} in the first batches.")
            continue

        input_tensor = image_tensor.unsqueeze(0).to(device)
        rgb_img = image_tensor.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))

        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax = axes.flat[i]
        ax.imshow(visualization)
        ax.set_title(f"True Class: {class_names[class_idx]}")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(file_name)
    plt.close()
    print(f"Saved Grad-CAM image to {file_name}")

def visualize_class_distribution(train_labels, classes):
    """Visualizes class distribution before and after handling"""

    # Calculate class counts
    class_counts = [train_labels.count(idx) for idx in range(len(classes))]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original distribution
    x = np.arange(len(classes))  # Create x-coordinates for bars
    ax1.bar(x, class_counts)
    ax1.set_title('Original Class Distribution')
    ax1.set_xticks(x)  # Set explicit x-tick positions
    ax1.set_xticklabels(classes, rotation=45, ha='right')  # Align labels
    ax1.set_ylabel('Number of Samples')

    # After weighting/balancing
    weighted_counts = [count * class_weights[i].item() for i, count in enumerate(class_counts)]
    ax2.bar(x, weighted_counts)
    ax2.set_title('Effective Distribution After Class Weighting')
    ax2.set_xticks(x)  # Set explicit x-tick positions
    ax2.set_xticklabels(classes, rotation=45, ha='right')  # Align labels
    ax2.set_ylabel('Weighted Sample Importance')

    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()


def create_dataset_summary(train_data, val_data, test_data, classes):
    """Creates detailed dataset visualization and table"""

    # Calculate sizes and percentages
    total_size = len(train_data) + len(val_data) + len(test_data)
    sizes = [len(train_data), len(val_data), len(test_data), total_size]
    percentages = [size / total_size * 100 for size in sizes]
    splits = ['Train', 'Validation', 'Test', 'Total']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])

    # Plot dataset split distribution (exclude total from pie chart)
    ax1.pie(sizes[:-1], labels=splits[:-1], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Dataset Split Distribution')

    # Create table data
    table_data = [[f"{size:,d} ({pct:.1f}%)" for size, pct in zip(sizes, percentages)]]

    # Create table
    table = ax2.table(cellText=table_data,
                      rowLabels=['Samples'],
                      colLabels=splits,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.5, 2)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('dataset_summary.png')
    plt.close()

def visualize_preprocessing_steps(image_path):
    """Visualizes each step of preprocessing pipeline for a single image"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Steps Visualization', fontsize=16)

    # Original Image
    original_img = Image.open(image_path).convert('RGB')
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')

    # Resized Image
    resize_transform = transforms.Resize((224, 224))
    resized_img = resize_transform(original_img)
    axes[0, 1].imshow(resized_img)
    axes[0, 1].set_title('Resized (224x224)')

    # Flipped Images
    h_flip = transforms.RandomHorizontalFlip(p=1)(resized_img)
    v_flip = transforms.RandomVerticalFlip(p=1)(resized_img)
    axes[0, 2].imshow(h_flip)
    axes[0, 2].set_title('Horizontal Flip')
    axes[1, 0].imshow(v_flip)
    axes[1, 0].set_title('Vertical Flip')

    # Normalized Image (fix the clipping warning)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalized_img = normalize(to_tensor(resized_img))
    # Rescale normalized image to [0,1] range for visualization
    normalized_img = (normalized_img - normalized_img.min()) / (normalized_img.max() - normalized_img.min())
    axes[1, 1].imshow(normalized_img.permute(1, 2, 0))
    axes[1, 1].set_title('Normalized')

    # All transformations combined
    all_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    final_img = all_transforms(original_img)
    # Rescale final image for visualization
    final_img = (final_img - final_img.min()) / (final_img.max() - final_img.min())
    axes[1, 2].imshow(final_img.permute(1, 2, 0))
    axes[1, 2].set_title('Final Transformed')

    # Remove axes
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('preprocessing_steps.png')
    plt.close()

class DementiaDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def load_data():
    data_dir = r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data"
    classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

    all_data = []
    all_labels = []

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith((".jpg", ".png")):
                all_data.append(os.path.join(class_dir, file_name))
                all_labels.append(idx)

    train_data, test_data, train_labels, test_labels = train_test_split(
        all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )


    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")
    return train_data, val_data, test_data, train_labels, val_labels, test_labels, classes

def create_data_loaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = DementiaDataset(train_data, train_labels, transform=transform)
    val_dataset = DementiaDataset(val_data, val_labels, transform=test_transform)
    test_dataset = DementiaDataset(test_data, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


class DementiaModel(nn.Module):
    def __init__(self, num_classes):
        super(DementiaModel, self).__init__()

        # Load pretrained Vision Transformer
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)

        # Replace the classifier head
        # ViT-B/16 has 768 hidden dimensions in the final layer
        self.model.heads = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, classes, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    accumulation_steps = 2
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    save_model_path = "resnet_alzheimer_final.pth"
    early_stopping = EarlyStopping(patience=5)
    best_model_wts = model.state_dict()
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    running_loss = 0.0
    running_corrects = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            scaler.scale(loss).backward()

            if (i % 1000) == 0:
                print(f"{i + 1}/{len(train_loader)}")

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = float(val_running_corrects) / len(val_loader.dataset)*100

        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        evaluate_model(model, val_loader, classes, set="val")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print(f"Best validation accuracy: {best_acc:.2f}%")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_model_path)
    return model

def class_count(train_labels, classes):

    class_counts = [train_labels.count(idx) for idx in range(len(classes))]
    total = sum(class_counts)
    class_weights = [count for count in class_counts]
    class_weights = [class_weight/total for class_weight in class_weights]
    class_weights = [1.0/class_weight for class_weight in class_weights]
    class_weights = torch.tensor(class_weights).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    return class_count, class_weights

class BalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2.0, class_thresholds=None):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)

        # Additional weighting for the smallest class (Moderate Dementia)
        self.class_thresholds = class_thresholds or {1: 5.0}  # Extra weight for class index 1 (Moderate)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Basic focal loss calculation
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Add class-specific weights
        class_weights = self.class_weights[targets]

        # Add extra weighting for specific classes
        extra_weights = torch.ones_like(targets, dtype=torch.float32)
        for class_idx, multiplier in self.class_thresholds.items():
            extra_weights[targets == class_idx] *= multiplier

        # Combine all weighting factors
        final_weights = focal_weight * class_weights * extra_weights.to(inputs.device)

        return (final_weights * ce_loss).mean()


def create_weighted_focal_loss(class_weights, alpha=0.25, gamma=2.0):
    # Increase gamma for more focus on hard examples
    gamma = 3.0

    # Calculate inverse class frequency for thresholding
    total_samples = class_weights.sum()
    class_frequencies = class_weights / total_samples

    # Set higher threshold for classes with very low frequency
    class_thresholds = {}
    for i, freq in enumerate(class_frequencies):
        if freq < 0.01:  # If class represents less than 1% of data
            class_thresholds[i] = 8.0  # Higher weight for very rare classes
        elif freq < 0.05:  # If class represents less than 5% of data
            class_thresholds[i] = 5.0  # Moderate weight for somewhat rare classes

    return BalancedFocalLoss(
        class_weights=class_weights,
        alpha=alpha,
        gamma=gamma,
        class_thresholds=class_thresholds
    )


def evaluate_model(model, test_loader, classes, set=""):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = (test_correct / total) * 100
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Confusion Matrix ({set} set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    # Save the confusion matrix plot
    plt.savefig(f'confusion_matrix_{set}.png')
    plt.close()

    return accuracy, cm

def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


if __name__ == '__main__':
    # Load and prepare data
    train_data, val_data, test_data, train_labels, val_labels, test_labels, classes = load_data()
    train_loader, val_loader, test_loader = create_data_loaders(train_data, train_labels, val_data, val_labels, test_data, test_labels)

    class_count, class_weights = class_count(train_labels, classes)
    # Initialize model and training components
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DementiaModel(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3, amsgrad=True)

    from torchsummary import summary

    # In your main block, after model = DementiaModel(...)
    print(f"\n--- Model Architecture Summary for {type(model.model).__name__} ---")
    summary(model, (3, 224, 224))
    print("--- End of Summary ---\n")

    # Train model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer,  classes, num_epochs=20)

    # In your ViT script's main block, after training...

    # 1. Define the correct target layer for ViT
    target_layer_vit = [model.model.encoder.layers[-1].ln_1]

    # 2. Call the visualization function with the reshape_transform
    generate_grad_cam_visualizations(trained_model,
                                     test_loader,
                                     device,
                                     target_layer_vit,
                                     classes,
                                     file_name="gradcam_vit.png",
                                     reshape_transform=reshape_transform_vit)

    # In your main block, after training...
    # Note the addition of 'classes' at the end.
    generate_roc_pr_curves(trained_model, test_loader, device, classes)

    sample_image = train_data[0]  # Get first training image
    visualize_preprocessing_steps(sample_image)

    # Visualize class distribution
    visualize_class_distribution(train_labels, classes)

    # Create dataset summary
    create_dataset_summary(train_data, val_data, test_data, classes)

    # Evaluate model
    print("Evaluating model on test set...")
    test_accuracy, confusion_mat = evaluate_model(trained_model, test_loader, classes, set="test")
    print("\nConfusion Matrix:")
    print(confusion_mat)


"""
Results (Non Weighted) :-


(alzheimers) C:/Users/aravs/PycharmProjects/Alzheimers/new_ablations>python ablation2_only_vit.py
Train size: 55319, Validation size: 13830, Test size: 17288
Class counts: [3202, 312, 43021, 8784]
Class weights: tensor([ 17.2764, 177.3045,   1.2859,   6.2977], device='cuda:0')
Epoch 1/20
1/1729
1001/1729
Train Loss: 0.6851, Train Accuracy: 0.7768, Val Loss: 0.6494, Val Accuracy: 77.77%
Overall Test Accuracy: 77.77%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      1.00      0.87     10756
Very mild Dementia       0.00      0.00      0.00      2196

          accuracy                           0.78     13830
         macro avg       0.19      0.25      0.22     13830
      weighted avg       0.60      0.78      0.68     13830

Epoch 2/20
1/1729
1001/1729
Train Loss: 0.6538, Train Accuracy: 0.7777, Val Loss: 0.6417, Val Accuracy: 77.77%
Overall Test Accuracy: 77.77%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      1.00      0.87     10756
Very mild Dementia       0.00      0.00      0.00      2196

          accuracy                           0.78     13830
         macro avg       0.19      0.25      0.22     13830
      weighted avg       0.60      0.78      0.68     13830

Epoch 3/20
1/1729
1001/1729
Train Loss: 0.6409, Train Accuracy: 0.7775, Val Loss: 0.6751, Val Accuracy: 77.77%
Overall Test Accuracy: 77.77%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      1.00      0.87     10756
Very mild Dementia       0.45      0.00      0.00      2196

          accuracy                           0.78     13830
         macro avg       0.31      0.25      0.22     13830
      weighted avg       0.68      0.78      0.68     13830

Epoch 4/20
1/1729
1001/1729
Train Loss: 0.6248, Train Accuracy: 0.7779, Val Loss: 0.6150, Val Accuracy: 78.00%
Overall Test Accuracy: 78.00%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      1.00      0.88     10756
Very mild Dementia       0.70      0.02      0.04      2196

          accuracy                           0.78     13830
         macro avg       0.37      0.26      0.23     13830
      weighted avg       0.72      0.78      0.69     13830

Epoch 5/20
1/1729
1001/1729
Train Loss: 0.6129, Train Accuracy: 0.7792, Val Loss: 0.5948, Val Accuracy: 78.04%
Overall Test Accuracy: 78.04%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      0.99      0.88     10756
Very mild Dementia       0.54      0.05      0.08      2196

          accuracy                           0.78     13830
         macro avg       0.33      0.26      0.24     13830
      weighted avg       0.69      0.78      0.69     13830

Epoch 6/20
1/1729
1001/1729
Train Loss: 0.5958, Train Accuracy: 0.7803, Val Loss: 0.5811, Val Accuracy: 78.39%
Overall Test Accuracy: 78.39%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.79      0.99      0.88     10756
Very mild Dementia       0.59      0.08      0.14      2196

          accuracy                           0.78     13830
         macro avg       0.34      0.27      0.26     13830
      weighted avg       0.71      0.78      0.71     13830

Epoch 7/20
1/1729
1001/1729
Train Loss: 0.5777, Train Accuracy: 0.7831, Val Loss: 0.5608, Val Accuracy: 78.53%
Overall Test Accuracy: 78.53%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.80      0.97      0.88     10756
Very mild Dementia       0.49      0.19      0.28      2196

          accuracy                           0.79     13830
         macro avg       0.32      0.29      0.29     13830
      weighted avg       0.70      0.79      0.73     13830

Epoch 8/20
1/1729
1001/1729
Train Loss: 0.5516, Train Accuracy: 0.7868, Val Loss: 0.5575, Val Accuracy: 78.31%
Overall Test Accuracy: 78.31%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      1.00      0.88     10756
Very mild Dementia       0.73      0.05      0.09      2196

          accuracy                           0.78     13830
         macro avg       0.38      0.26      0.24     13830
      weighted avg       0.73      0.78      0.70     13830

Epoch 9/20
1/1729
1001/1729
Train Loss: 0.5183, Train Accuracy: 0.7918, Val Loss: 0.4909, Val Accuracy: 79.49%
Overall Test Accuracy: 79.49%
                    precision    recall  f1-score   support

     Mild Dementia       0.56      0.08      0.14       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.88      0.91      0.90     10756
Very mild Dementia       0.44      0.52      0.48      2196

          accuracy                           0.79     13830
         macro avg       0.47      0.38      0.38     13830
      weighted avg       0.79      0.79      0.78     13830

Epoch 10/20
1/1729
1001/1729
Train Loss: 0.4881, Train Accuracy: 0.7980, Val Loss: 0.4811, Val Accuracy: 80.56%
Overall Test Accuracy: 80.56%
                    precision    recall  f1-score   support

     Mild Dementia       0.52      0.12      0.20       800
 Moderate Dementia       0.29      0.17      0.21        78
      Non Demented       0.85      0.95      0.90     10756
Very mild Dementia       0.52      0.35      0.42      2196

          accuracy                           0.81     13830
         macro avg       0.54      0.40      0.43     13830
      weighted avg       0.77      0.81      0.78     13830

Epoch 11/20
1/1729
1001/1729
Train Loss: 0.4566, Train Accuracy: 0.8120, Val Loss: 0.4221, Val Accuracy: 82.34%
Overall Test Accuracy: 82.34%
                    precision    recall  f1-score   support

     Mild Dementia       0.55      0.39      0.46       800
 Moderate Dementia       0.88      0.18      0.30        78
      Non Demented       0.85      0.97      0.91     10756
Very mild Dementia       0.63      0.28      0.39      2196

          accuracy                           0.82     13830
         macro avg       0.73      0.46      0.51     13830
      weighted avg       0.80      0.82      0.80     13830

Epoch 12/20
1/1729
1001/1729
Train Loss: 0.5846, Train Accuracy: 0.7865, Val Loss: 0.5648, Val Accuracy: 78.29%
Overall Test Accuracy: 78.29%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       1.00      0.06      0.12        78
      Non Demented       0.79      1.00      0.88     10756
Very mild Dementia       0.54      0.05      0.09      2196

          accuracy                           0.78     13830
         macro avg       0.58      0.28      0.27     13830
      weighted avg       0.70      0.78      0.70     13830

Epoch 13/20
1/1729
1001/1729
Train Loss: 0.5284, Train Accuracy: 0.7924, Val Loss: 0.5347, Val Accuracy: 79.03%
Overall Test Accuracy: 79.03%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.50      0.01      0.03        78
      Non Demented       0.86      0.92      0.89     10756
Very mild Dementia       0.44      0.45      0.44      2196

          accuracy                           0.79     13830
         macro avg       0.45      0.35      0.34     13830
      weighted avg       0.74      0.79      0.76     13830

Epoch 14/20
1/1729
1001/1729
Train Loss: 0.4583, Train Accuracy: 0.8139, Val Loss: 0.4151, Val Accuracy: 82.51%
Overall Test Accuracy: 82.51%
                    precision    recall  f1-score   support

     Mild Dementia       0.60      0.26      0.36       800
 Moderate Dementia       0.60      0.36      0.45        78
      Non Demented       0.89      0.94      0.91     10756
Very mild Dementia       0.53      0.51      0.52      2196

          accuracy                           0.83     13830
         macro avg       0.65      0.52      0.56     13830
      weighted avg       0.81      0.83      0.81     13830

Epoch 15/20
1/1729
1001/1729
Train Loss: 0.4200, Train Accuracy: 0.8253, Val Loss: 0.4160, Val Accuracy: 83.11%
Overall Test Accuracy: 83.11%
                    precision    recall  f1-score   support

     Mild Dementia       0.66      0.30      0.41       800
 Moderate Dementia       0.28      0.55      0.37        78
      Non Demented       0.87      0.96      0.91     10756
Very mild Dementia       0.60      0.41      0.49      2196

          accuracy                           0.83     13830
         macro avg       0.60      0.55      0.54     13830
      weighted avg       0.81      0.83      0.81     13830

Epoch 16/20
1/1729
1001/1729
Train Loss: 0.3955, Train Accuracy: 0.8361, Val Loss: 0.3928, Val Accuracy: 83.64%
Overall Test Accuracy: 83.64%
                    precision    recall  f1-score   support

     Mild Dementia       0.73      0.27      0.40       800
 Moderate Dementia       0.62      0.42      0.50        78
      Non Demented       0.87      0.97      0.92     10756
Very mild Dementia       0.61      0.40      0.48      2196

          accuracy                           0.84     13830
         macro avg       0.71      0.52      0.58     13830
      weighted avg       0.82      0.84      0.81     13830

Epoch 17/20
1/1729
1001/1729
Train Loss: 0.3712, Train Accuracy: 0.8454, Val Loss: 0.3483, Val Accuracy: 85.27%
Overall Test Accuracy: 85.27%
                    precision    recall  f1-score   support

     Mild Dementia       0.79      0.38      0.51       800
 Moderate Dementia       0.57      0.58      0.57        78
      Non Demented       0.89      0.97      0.93     10756
Very mild Dementia       0.64      0.47      0.54      2196

          accuracy                           0.85     13830
         macro avg       0.72      0.60      0.64     13830
      weighted avg       0.84      0.85      0.84     13830

Epoch 18/20
1/1729
1001/1729
Train Loss: 0.3826, Train Accuracy: 0.8430, Val Loss: 0.3445, Val Accuracy: 85.73%
Overall Test Accuracy: 85.73%
                    precision    recall  f1-score   support

     Mild Dementia       0.62      0.64      0.63       800
 Moderate Dementia       0.71      0.47      0.57        78
      Non Demented       0.90      0.96      0.93     10756
Very mild Dementia       0.69      0.47      0.56      2196

          accuracy                           0.86     13830
         macro avg       0.73      0.63      0.67     13830
      weighted avg       0.85      0.86      0.85     13830

Epoch 19/20
1/1729
1001/1729
Train Loss: 0.3359, Train Accuracy: 0.8625, Val Loss: 0.3219, Val Accuracy: 86.77%
Overall Test Accuracy: 86.77%
                    precision    recall  f1-score   support

     Mild Dementia       0.69      0.64      0.66       800
 Moderate Dementia       1.00      0.27      0.42        78
      Non Demented       0.93      0.93      0.93     10756
Very mild Dementia       0.62      0.65      0.64      2196

          accuracy                           0.87     13830
         macro avg       0.81      0.62      0.66     13830
      weighted avg       0.87      0.87      0.87     13830

Epoch 20/20
1/1729
1001/1729
Train Loss: 0.3140, Train Accuracy: 0.8713, Val Loss: 0.3174, Val Accuracy: 86.98%
Overall Test Accuracy: 86.98%
                    precision    recall  f1-score   support

     Mild Dementia       0.64      0.69      0.66       800
 Moderate Dementia       0.92      0.46      0.62        78
      Non Demented       0.93      0.94      0.93     10756
Very mild Dementia       0.65      0.62      0.64      2196

          accuracy                           0.87     13830
         macro avg       0.79      0.68      0.71     13830
      weighted avg       0.87      0.87      0.87     13830

Best validation accuracy: 86.98%
Evaluating model on test set...
Overall Test Accuracy: 86.41%
                    precision    recall  f1-score   support

     Mild Dementia       0.61      0.66      0.64      1000
 Moderate Dementia       0.83      0.45      0.58        98
      Non Demented       0.93      0.93      0.93     13445
Very mild Dementia       0.63      0.62      0.63      2745

          accuracy                           0.86     17288
         macro avg       0.75      0.67      0.69     17288
      weighted avg       0.87      0.86      0.86     17288


Confusion Matrix:
[[  664     1   100   235]
 [   20    44    13    21]
 [  178     0 12517   750]
 [  225     8   798  1714]]
"""





"""
Results (Weighted) :-

(alzheimers) C:/Users/aravs/PycharmProjects/Alzheimers/new_ablations>python ablation2_only_vit.py
Train size: 55319, Validation size: 13830, Test size: 17288
Class counts: [3202, 312, 43021, 8784]
Class weights: tensor([ 17.2764, 177.3045,   1.2859,   6.2977], device='cuda:0')
Epoch 1/20
1/1729
1001/1729
Train Loss: 1.3538, Train Accuracy: 0.4427, Val Loss: 1.3045, Val Accuracy: 77.40%
Overall Test Accuracy: 77.40%
                    precision    recall  f1-score   support

     Mild Dementia       0.21      0.02      0.04       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.78      0.99      0.87     10756
Very mild Dementia       0.28      0.02      0.03      2196

          accuracy                           0.77     13830
         macro avg       0.32      0.26      0.24     13830
      weighted avg       0.67      0.77      0.69     13830

Epoch 2/20
1/1729
1001/1729
Train Loss: 1.3229, Train Accuracy: 0.5350, Val Loss: 1.2883, Val Accuracy: 75.87%
Overall Test Accuracy: 75.87%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.79      0.96      0.86     10756
Very mild Dementia       0.26      0.09      0.14      2196

          accuracy                           0.76     13830
         macro avg       0.26      0.26      0.25     13830
      weighted avg       0.65      0.76      0.69     13830

Epoch 3/20
1/1729
1001/1729
Train Loss: 1.3087, Train Accuracy: 0.5874, Val Loss: 1.3090, Val Accuracy: 60.87%
Overall Test Accuracy: 60.87%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.00      0.00      0.00        78
      Non Demented       0.80      0.71      0.75     10756
Very mild Dementia       0.19      0.38      0.25      2196

          accuracy                           0.61     13830
         macro avg       0.25      0.27      0.25     13830
      weighted avg       0.65      0.61      0.62     13830

Epoch 4/20
1/1729
1001/1729
Train Loss: 1.3008, Train Accuracy: 0.6352, Val Loss: 1.2634, Val Accuracy: 67.97%
Overall Test Accuracy: 67.97%
                    precision    recall  f1-score   support

     Mild Dementia       0.08      0.07      0.07       800
 Moderate Dementia       0.03      0.56      0.06        78
      Non Demented       0.80      0.86      0.83     10756
Very mild Dementia       0.16      0.02      0.04      2196

          accuracy                           0.68     13830
         macro avg       0.27      0.38      0.25     13830
      weighted avg       0.65      0.68      0.65     13830

Epoch 5/20
1/1729
1001/1729
Train Loss: 1.2914, Train Accuracy: 0.6385, Val Loss: 1.2739, Val Accuracy: 68.15%
Overall Test Accuracy: 68.15%
                    precision    recall  f1-score   support

     Mild Dementia       0.16      0.01      0.02       800
 Moderate Dementia       0.03      0.62      0.05        78
      Non Demented       0.80      0.87      0.83     10756
Very mild Dementia       0.19      0.02      0.03      2196

          accuracy                           0.68     13830
         macro avg       0.29      0.38      0.23     13830
      weighted avg       0.66      0.68      0.65     13830

Epoch 6/20
1/1729
1001/1729
Train Loss: 1.2907, Train Accuracy: 0.6554, Val Loss: 1.2688, Val Accuracy: 71.52%
Overall Test Accuracy: 71.52%
                    precision    recall  f1-score   support

     Mild Dementia       0.10      0.19      0.13       800
 Moderate Dementia       0.04      0.01      0.02        78
      Non Demented       0.80      0.91      0.85     10756
Very mild Dementia       0.00      0.00      0.00      2196

          accuracy                           0.72     13830
         macro avg       0.23      0.28      0.25     13830
      weighted avg       0.62      0.72      0.67     13830

Epoch 7/20
1/1729
1001/1729
Train Loss: 1.2838, Train Accuracy: 0.6490, Val Loss: 1.2897, Val Accuracy: 61.79%
Overall Test Accuracy: 61.79%
                    precision    recall  f1-score   support

     Mild Dementia       0.12      0.00      0.01       800
 Moderate Dementia       0.02      0.65      0.04        78
      Non Demented       0.80      0.77      0.79     10756
Very mild Dementia       0.16      0.09      0.11      2196

          accuracy                           0.62     13830
         macro avg       0.28      0.38      0.24     13830
      weighted avg       0.66      0.62      0.63     13830

Epoch 8/20
1/1729
1001/1729
Train Loss: 1.2906, Train Accuracy: 0.6651, Val Loss: 1.2800, Val Accuracy: 65.81%
Overall Test Accuracy: 65.81%
                    precision    recall  f1-score   support

     Mild Dementia       0.11      0.10      0.10       800
 Moderate Dementia       0.08      0.03      0.04        78
      Non Demented       0.80      0.80      0.80     10756
Very mild Dementia       0.19      0.21      0.20      2196

          accuracy                           0.66     13830
         macro avg       0.29      0.28      0.28     13830
      weighted avg       0.66      0.66      0.66     13830

Epoch 9/20
1/1729
1001/1729
Train Loss: 1.2722, Train Accuracy: 0.6587, Val Loss: 1.2459, Val Accuracy: 71.84%
Overall Test Accuracy: 71.84%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.06      0.41      0.11        78
      Non Demented       0.79      0.90      0.84     10756
Very mild Dementia       0.23      0.12      0.16      2196

          accuracy                           0.72     13830
         macro avg       0.27      0.36      0.28     13830
      weighted avg       0.65      0.72      0.68     13830

Epoch 10/20
1/1729
1001/1729
Train Loss: 1.2737, Train Accuracy: 0.6763, Val Loss: 1.2440, Val Accuracy: 75.52%
Overall Test Accuracy: 75.52%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.12      0.38      0.18        78
      Non Demented       0.79      0.96      0.86     10756
Very mild Dementia       0.26      0.06      0.10      2196

          accuracy                           0.76     13830
         macro avg       0.29      0.35      0.29     13830
      weighted avg       0.65      0.76      0.69     13830

Epoch 11/20
1/1729
1001/1729
Train Loss: 1.2645, Train Accuracy: 0.6489, Val Loss: 1.2287, Val Accuracy: 67.51%
Overall Test Accuracy: 67.51%
                    precision    recall  f1-score   support

     Mild Dementia       0.09      0.07      0.08       800
 Moderate Dementia       0.08      0.37      0.13        78
      Non Demented       0.80      0.83      0.81     10756
Very mild Dementia       0.21      0.16      0.18      2196

          accuracy                           0.68     13830
         macro avg       0.29      0.36      0.30     13830
      weighted avg       0.66      0.68      0.67     13830

Epoch 12/20
1/1729
1001/1729
Train Loss: 1.2759, Train Accuracy: 0.6088, Val Loss: 1.2465, Val Accuracy: 64.19%
Overall Test Accuracy: 64.19%
                    precision    recall  f1-score   support

     Mild Dementia       0.07      0.15      0.10       800
 Moderate Dementia       0.04      0.62      0.08        78
      Non Demented       0.80      0.81      0.81     10756
Very mild Dementia       0.00      0.00      0.00      2196

          accuracy                           0.64     13830
         macro avg       0.23      0.39      0.24     13830
      weighted avg       0.63      0.64      0.63     13830

Epoch 13/20
1/1729
1001/1729
Train Loss: 1.2725, Train Accuracy: 0.6414, Val Loss: 1.2334, Val Accuracy: 71.76%
Overall Test Accuracy: 71.76%
                    precision    recall  f1-score   support

     Mild Dementia       0.07      0.00      0.01       800
 Moderate Dementia       0.05      0.62      0.09        78
      Non Demented       0.79      0.91      0.85     10756
Very mild Dementia       0.21      0.05      0.08      2196

          accuracy                           0.72     13830
         macro avg       0.28      0.39      0.26     13830
      weighted avg       0.65      0.72      0.67     13830

Epoch 14/20
1/1729
1001/1729
Train Loss: 1.2402, Train Accuracy: 0.6165, Val Loss: 1.2155, Val Accuracy: 73.91%
Overall Test Accuracy: 73.91%
                    precision    recall  f1-score   support

     Mild Dementia       0.06      0.00      0.00       800
 Moderate Dementia       0.09      0.47      0.14        78
      Non Demented       0.79      0.94      0.85     10756
Very mild Dementia       0.20      0.05      0.08      2196

          accuracy                           0.74     13830
         macro avg       0.28      0.37      0.27     13830
      weighted avg       0.65      0.74      0.68     13830

Epoch 15/20
1/1729
1001/1729
Train Loss: 1.2398, Train Accuracy: 0.6067, Val Loss: 1.2216, Val Accuracy: 34.37%
Overall Test Accuracy: 34.37%
                    precision    recall  f1-score   support

     Mild Dementia       0.07      0.06      0.06       800
 Moderate Dementia       0.10      0.54      0.17        78
      Non Demented       0.80      0.30      0.44     10756
Very mild Dementia       0.16      0.64      0.26      2196

          accuracy                           0.34     13830
         macro avg       0.28      0.38      0.23     13830
      weighted avg       0.65      0.34      0.39     13830

Epoch 16/20
1/1729
1001/1729
Train Loss: 1.2221, Train Accuracy: 0.6369, Val Loss: 1.1937, Val Accuracy: 65.63%
Overall Test Accuracy: 65.63%
                    precision    recall  f1-score   support

     Mild Dementia       0.08      0.03      0.04       800
 Moderate Dementia       0.16      0.50      0.24        78
      Non Demented       0.79      0.80      0.80     10756
Very mild Dementia       0.16      0.17      0.16      2196

          accuracy                           0.66     13830
         macro avg       0.30      0.38      0.31     13830
      weighted avg       0.64      0.66      0.65     13830

Epoch 17/20
1/1729
1001/1729
Train Loss: 1.2122, Train Accuracy: 0.5676, Val Loss: 1.1875, Val Accuracy: 72.10%
Overall Test Accuracy: 72.10%
                    precision    recall  f1-score   support

     Mild Dementia       0.11      0.05      0.07       800
 Moderate Dementia       0.09      0.63      0.15        78
      Non Demented       0.79      0.91      0.84     10756
Very mild Dementia       0.18      0.03      0.06      2196

          accuracy                           0.72     13830
         macro avg       0.29      0.41      0.28     13830
      weighted avg       0.65      0.72      0.67     13830

Epoch 18/20
1/1729
1001/1729
Train Loss: 1.2068, Train Accuracy: 0.6016, Val Loss: 1.2005, Val Accuracy: 66.78%
Overall Test Accuracy: 66.78%
                    precision    recall  f1-score   support

     Mild Dementia       0.07      0.05      0.06       800
 Moderate Dementia       0.07      0.62      0.13        78
      Non Demented       0.79      0.83      0.81     10756
Very mild Dementia       0.18      0.11      0.13      2196

          accuracy                           0.67     13830
         macro avg       0.28      0.40      0.28     13830
      weighted avg       0.65      0.67      0.65     13830

Epoch 19/20
1/1729
1001/1729
Train Loss: 1.2074, Train Accuracy: 0.5740, Val Loss: 1.1895, Val Accuracy: 70.82%
Overall Test Accuracy: 70.82%
                    precision    recall  f1-score   support

     Mild Dementia       0.04      0.03      0.04       800
 Moderate Dementia       0.16      0.53      0.24        78
      Non Demented       0.78      0.90      0.83     10756
Very mild Dementia       0.15      0.04      0.06      2196

          accuracy                           0.71     13830
         macro avg       0.28      0.37      0.29     13830
      weighted avg       0.63      0.71      0.66     13830

Epoch 20/20
1/1729
1001/1729
Train Loss: 1.2350, Train Accuracy: 0.5724, Val Loss: 1.2002, Val Accuracy: 75.87%
Overall Test Accuracy: 75.87%
                    precision    recall  f1-score   support

     Mild Dementia       0.08      0.01      0.03       800
 Moderate Dementia       0.12      0.49      0.19        78
      Non Demented       0.78      0.97      0.87     10756
Very mild Dementia       0.00      0.00      0.00      2196

          accuracy                           0.76     13830
         macro avg       0.25      0.37      0.27     13830
      weighted avg       0.61      0.76      0.68     13830

Best validation accuracy: 77.40%
Evaluating model on test set...
Overall Test Accuracy: 75.96%
                    precision    recall  f1-score   support

     Mild Dementia       0.06      0.01      0.02      1000
 Moderate Dementia       0.10      0.38      0.16        98
      Non Demented       0.78      0.97      0.87     13445
Very mild Dementia       0.00      0.00      0.00      2745

          accuracy                           0.76     17288
         macro avg       0.24      0.34      0.26     17288
      weighted avg       0.61      0.76      0.68     17288


Confusion Matrix:
[[   10    33   956     1]
 [    5    37    56     0]
 [  120   239 13085     1]
 [   35    47  2663     0]]

"""