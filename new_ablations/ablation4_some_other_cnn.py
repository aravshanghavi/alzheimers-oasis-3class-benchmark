#needs model change to some other CNN -> Loss func remains same -> change made; need to test -> done; results pasted below


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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import seaborn as sns


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

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=weights)

        # Remove the original classifier
        self.model.classifier = nn.Identity()

        # Add custom classifier
        # EfficientNet-B0 outputs 1280 features from avgpool
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features using EfficientNet backbone
        x = self.model(x)  # This will output (batch_size, 1280) after avgpool
        x = self.classifier(x)
        return x


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

"""class WeightedFocalLossMultiClass(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2):
        super(WeightedFocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights.to(inputs.device), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return nn.CrossEntropyLoss(weight=class_weights)

def create_weighted_focal_loss(class_weights, alpha=0.25, gamma=2):
    return nn.CrossEntropyLoss(weight=class_weights)"""


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


if __name__ == '__main__':
    # Load and prepare data
    train_data, val_data, test_data, train_labels, val_labels, test_labels, classes = load_data()
    train_loader, val_loader, test_loader = create_data_loaders(train_data, train_labels, val_data, val_labels, test_data, test_labels)

    class_count, class_weights = class_count(train_labels, classes)
    # Initialize model and training components
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DementiaModel(num_classes=len(classes))
    criterion = create_weighted_focal_loss(class_weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3, amsgrad=True)

    # Train model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer,  classes, num_epochs=20)

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
Results :-

(alzheimers) C:\Users\aravs\PycharmProjects\Alzheimers\new_ablations>python ablation4_some_other_cnn.py
Train size: 55319, Validation size: 13830, Test size: 17288
Class counts: [3202, 312, 43021, 8784]
Class weights: tensor([ 17.2764, 177.3045,   1.2859,   6.2977], device='cuda:0')
Epoch 1/20
1/1729
1001/1729
Train Loss: 0.9958, Train Accuracy: 0.7014, Val Loss: 0.7657, Val Accuracy: 73.49%
Overall Test Accuracy: 73.49%
                    precision    recall  f1-score   support

     Mild Dementia       0.00      0.00      0.00       800
 Moderate Dementia       0.09      1.00      0.16        78
      Non Demented       0.97      0.77      0.86     10756
Very mild Dementia       0.41      0.81      0.54      2196

          accuracy                           0.73     13830
         macro avg       0.37      0.65      0.39     13830
      weighted avg       0.82      0.73      0.76     13830

Epoch 2/20
1/1729
1001/1729
Train Loss: 0.5603, Train Accuracy: 0.7583, Val Loss: 0.3241, Val Accuracy: 82.85%
Overall Test Accuracy: 82.85%
                    precision    recall  f1-score   support

     Mild Dementia       0.80      0.68      0.74       800
 Moderate Dementia       0.59      0.90      0.71        78
      Non Demented       0.96      0.85      0.90     10756
Very mild Dementia       0.50      0.80      0.61      2196

          accuracy                           0.83     13830
         macro avg       0.71      0.81      0.74     13830
      weighted avg       0.87      0.83      0.84     13830

Epoch 3/20
1/1729
1001/1729
Train Loss: 0.3163, Train Accuracy: 0.8329, Val Loss: 0.2356, Val Accuracy: 86.04%
Overall Test Accuracy: 86.04%
                    precision    recall  f1-score   support

     Mild Dementia       0.85      0.62      0.72       800
 Moderate Dementia       0.33      1.00      0.49        78
      Non Demented       0.98      0.87      0.92     10756
Very mild Dementia       0.57      0.91      0.70      2196

          accuracy                           0.86     13830
         macro avg       0.68      0.85      0.71     13830
      weighted avg       0.90      0.86      0.87     13830

Epoch 4/20
1/1729
1001/1729
Train Loss: 0.2228, Train Accuracy: 0.8826, Val Loss: 0.0795, Val Accuracy: 94.23%
Overall Test Accuracy: 94.23%
                    precision    recall  f1-score   support

     Mild Dementia       0.95      0.97      0.96       800
 Moderate Dementia       0.90      1.00      0.95        78
      Non Demented       0.99      0.94      0.96     10756
Very mild Dementia       0.77      0.94      0.85      2196

          accuracy                           0.94     13830
         macro avg       0.90      0.96      0.93     13830
      weighted avg       0.95      0.94      0.94     13830

Epoch 5/20
1/1729
1001/1729
Train Loss: 0.1514, Train Accuracy: 0.9189, Val Loss: 0.0578, Val Accuracy: 97.14%
Overall Test Accuracy: 97.14%
                    precision    recall  f1-score   support

     Mild Dementia       0.92      0.97      0.94       800
 Moderate Dementia       0.74      1.00      0.85        78
      Non Demented       0.99      0.98      0.98     10756
Very mild Dementia       0.92      0.94      0.93      2196

          accuracy                           0.97     13830
         macro avg       0.89      0.97      0.93     13830
      weighted avg       0.97      0.97      0.97     13830

Epoch 6/20
1/1729
1001/1729
Train Loss: 0.0578, Train Accuracy: 0.9657, Val Loss: 0.0253, Val Accuracy: 97.94%
Overall Test Accuracy: 97.94%
                    precision    recall  f1-score   support

     Mild Dementia       0.97      1.00      0.98       800
 Moderate Dementia       0.97      1.00      0.99        78
      Non Demented       1.00      0.98      0.99     10756
Very mild Dementia       0.90      0.99      0.94      2196

          accuracy                           0.98     13830
         macro avg       0.96      0.99      0.98     13830
      weighted avg       0.98      0.98      0.98     13830

Epoch 7/20
1/1729
1001/1729
Train Loss: 0.0324, Train Accuracy: 0.9817, Val Loss: 0.0093, Val Accuracy: 99.33%
Overall Test Accuracy: 99.33%
                    precision    recall  f1-score   support

     Mild Dementia       0.99      1.00      0.99       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      0.99      1.00     10756
Very mild Dementia       0.97      1.00      0.98      2196

          accuracy                           0.99     13830
         macro avg       0.99      1.00      0.99     13830
      weighted avg       0.99      0.99      0.99     13830

Epoch 8/20
1/1729
1001/1729
Train Loss: 0.0636, Train Accuracy: 0.9670, Val Loss: 0.0124, Val Accuracy: 99.16%
Overall Test Accuracy: 99.16%
                    precision    recall  f1-score   support

     Mild Dementia       0.99      0.99      0.99       800
 Moderate Dementia       0.96      1.00      0.98        78
      Non Demented       1.00      0.99      0.99     10756
Very mild Dementia       0.96      0.99      0.98      2196

          accuracy                           0.99     13830
         macro avg       0.98      0.99      0.99     13830
      weighted avg       0.99      0.99      0.99     13830

Epoch 9/20
1/1729
1001/1729
Train Loss: 0.0260, Train Accuracy: 0.9854, Val Loss: 0.0038, Val Accuracy: 99.73%
Overall Test Accuracy: 99.73%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       0.97      1.00      0.99        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       0.98      1.00      0.99      2196

          accuracy                           1.00     13830
         macro avg       0.99      1.00      0.99     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 10/20
1/1729
1001/1729
Train Loss: 0.0156, Train Accuracy: 0.9912, Val Loss: 0.0037, Val Accuracy: 99.83%
Overall Test Accuracy: 99.83%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       0.99      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 11/20
1/1729
1001/1729
Train Loss: 0.0216, Train Accuracy: 0.9905, Val Loss: 0.0027, Val Accuracy: 99.89%
Overall Test Accuracy: 99.89%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       0.99      1.00      0.99        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 12/20
1/1729
1001/1729
Train Loss: 0.0198, Train Accuracy: 0.9915, Val Loss: 0.0048, Val Accuracy: 99.72%
Overall Test Accuracy: 99.72%
                    precision    recall  f1-score   support

     Mild Dementia       0.99      1.00      1.00       800
 Moderate Dementia       0.94      1.00      0.97        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       0.99      1.00      0.99      2196

          accuracy                           1.00     13830
         macro avg       0.98      1.00      0.99     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 13/20
1/1729
1001/1729
Train Loss: 0.0160, Train Accuracy: 0.9931, Val Loss: 0.0019, Val Accuracy: 99.92%
Overall Test Accuracy: 99.92%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 14/20
1/1729
1001/1729
Train Loss: 0.0078, Train Accuracy: 0.9962, Val Loss: 0.0011, Val Accuracy: 99.92%
Overall Test Accuracy: 99.92%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 15/20
1/1729
1001/1729
Train Loss: 0.0054, Train Accuracy: 0.9973, Val Loss: 0.0008, Val Accuracy: 99.95%
Overall Test Accuracy: 99.95%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 16/20
1/1729
1001/1729
Train Loss: 0.0031, Train Accuracy: 0.9982, Val Loss: 0.0003, Val Accuracy: 99.99%
Overall Test Accuracy: 99.99%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 17/20
1/1729
1001/1729
Train Loss: 0.0028, Train Accuracy: 0.9985, Val Loss: 0.0005, Val Accuracy: 99.99%
Overall Test Accuracy: 99.99%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 18/20
1/1729
1001/1729
Train Loss: 0.0042, Train Accuracy: 0.9982, Val Loss: 0.0004, Val Accuracy: 99.99%
Overall Test Accuracy: 99.99%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 19/20
1/1729
1001/1729
Train Loss: 0.0031, Train Accuracy: 0.9985, Val Loss: 0.0004, Val Accuracy: 99.97%
Overall Test Accuracy: 99.97%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      1.00      1.00       800
 Moderate Dementia       1.00      1.00      1.00        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      1.00      1.00      2196

          accuracy                           1.00     13830
         macro avg       1.00      1.00      1.00     13830
      weighted avg       1.00      1.00      1.00     13830

Epoch 20/20
1/1729
1001/1729
Train Loss: 0.0029, Train Accuracy: 0.9988, Val Loss: 0.0153, Val Accuracy: 99.56%
Overall Test Accuracy: 99.56%
                    precision    recall  f1-score   support

     Mild Dementia       1.00      0.98      0.99       800
 Moderate Dementia       0.72      1.00      0.83        78
      Non Demented       1.00      1.00      1.00     10756
Very mild Dementia       1.00      0.99      0.99      2196

          accuracy                           1.00     13830
         macro avg       0.93      0.99      0.95     13830
      weighted avg       1.00      1.00      1.00     13830

Best validation accuracy: 99.99%
Evaluating model on test set...
Overall Test Accuracy: 99.57%
                    precision    recall  f1-score   support

     Mild Dementia       0.99      0.98      0.99      1000
 Moderate Dementia       0.76      1.00      0.86        98
      Non Demented       1.00      1.00      1.00     13445
Very mild Dementia       0.99      0.99      0.99      2745

          accuracy                           1.00     17288
         macro avg       0.94      0.99      0.96     17288
      weighted avg       1.00      1.00      1.00     17288


Confusion Matrix:
[[  980    19     0     1]
 [    0    98     0     0]
 [    3     4 13419    19]
 [    2     8    19  2716]]

"""