# ================================================================================
# DEFINITIVE K-FOLD SCRIPT FOR EFFICIENTNET-B0 (Corrected Model)
# This final version uses the user-specified DementiaModel class structure
# and the robust K-Fold evaluation pipeline.
# ================================================================================

import os
import re
import random
import copy
from collections import defaultdict
from itertools import cycle
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image

# --- Suppress Warnings ---
warnings.filterwarnings('ignore')

# --- Import Grad-CAM and its utilities ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ================================================================================
# SECTION 1: MODEL DEFINITION (EfficientNet-B0 - User's Corrected Version)
# ================================================================================

class DementiaModel(nn.Module):
    def __init__(self, num_classes):
        super(DementiaModel, self).__init__()
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)

        # Remove the original classifier by replacing it with an identity layer
        self.model.classifier = nn.Identity()

        # Add a new, separate custom classifier
        # EfficientNet-B0's feature extractor outputs 1280 features
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features using the backbone
        features = self.model(x)
        # Pass features through the new classifier
        output = self.classifier(features)
        return output


# ================================================================================
# SECTION 2: DATA HANDLING & HELPERS
# ================================================================================

class DementiaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_subject_data(data_dir, classes):
    """Groups all images by subject and assigns a label to each subject."""
    subject_pattern = re.compile(r'(OAS\d_\d{4}_MR\d)')
    subjects_to_images = defaultdict(list)
    subject_to_label = {}
    print("Scanning data directory...")
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir): continue
        for file_name in os.listdir(class_dir):
            match = subject_pattern.match(file_name)
            if match:
                subject_id = match.group(1)
                subjects_to_images[subject_id].append(os.path.join(class_dir, file_name))
                if subject_id not in subject_to_label:
                    subject_to_label[subject_id] = label_idx
    subject_list = list(subjects_to_images.keys())
    labels_list = [subject_to_label[s] for s in subject_list]
    print(f"Found {len(subject_list)} unique subjects across {len(classes)} classes.")
    return subject_list, labels_list, subjects_to_images, subject_to_label


def unpack_split_data(subject_list, subjects_to_images, subject_to_label):
    images, labels = [], []
    for subj in subject_list:
        subj_images = subjects_to_images[subj]
        images.extend(subj_images)
        labels.extend([subject_to_label[subj]] * len(subj_images))
    return images, labels


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("  Early stopping triggered.")
                self.early_stop = True


# ================================================================================
# SECTION 3: CORE TRAINING AND EVALUATION LOGIC
# ================================================================================

def train_and_validate_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"  Epoch {epoch + 1}/{num_epochs} -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    model.load_state_dict(best_model_wts)
    return model, best_acc.item()


def final_evaluation(model, test_loader, device, class_names, file_prefix=""):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    y_true, y_pred, y_proba = np.array(y_true), np.array(y_pred), np.array(y_proba)

    print("\nFinal Test Set Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\n--- Generating Final Visualizations ---")
    generate_roc_pr_curves(y_true, y_proba, class_names, file_prefix=file_prefix)

    # Grad-CAM
    try:
        target_layer = [model.model.features[-1]]  # Correct for your CNN model structure
        generate_grad_cam_visualizations(model, test_loader, device, target_layer, class_names, file_prefix=file_prefix)
    except Exception as e:
        print(f"Could not generate Grad-CAM visualizations. Error: {e}")


def generate_roc_pr_curves(y_true, y_probas, class_names, file_prefix=""):
    n_classes = len(class_names)
    # ROC
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        y_true_class = (y_true == i)
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC of class {class_names[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f'{file_prefix} ROC Curves')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.legend(loc="lower right")
    plt.savefig(f"{file_prefix}_roc_curves.png");
    plt.close()

    # PR
    precision, recall, avg_precision = dict(), dict(), dict()
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        y_true_class = (y_true == i)
        precision[i], recall[i], _ = precision_recall_curve(y_true_class, y_probas[:, i])
        avg_precision[i] = average_precision_score(y_true_class, y_probas[:, i])
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR of class {class_names[i]} (AP = {avg_precision[i]:0.2f})')
    plt.title(f'{file_prefix} Precision-Recall Curves');
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.legend(loc="best")
    plt.savefig(f"{file_prefix}_pr_curves.png");
    plt.close()


def generate_grad_cam_visualizations(model, test_loader, device, target_layers, class_names,
                                     file_name="gradcam_heatmaps.png"):
    """Generates and saves Grad-CAM visualizations for one sample from each class."""
    print("Generating Grad-CAM visualizations...")
    model.eval()

    # Get the number of classes from the provided list
    num_classes = len(class_names)

    # Initialize a dictionary to store one image per class
    images_per_class = {i: None for i in range(num_classes)}

    # Find one image for each class from the test set
    for images, labels in test_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if images_per_class[label] is None:
                images_per_class[label] = images[i]
        # Stop once we have one image for every class
        if all(v is not None for v in images_per_class.values()):
            break

    cam = GradCAM(model=model, target_layers=target_layers)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    if num_classes > 4:  # Adjust layout for more classes if needed
        fig, axes = plt.subplots((num_classes + 1) // 2, 2, figsize=(12, 6 * ((num_classes + 1) // 2)))

    fig.suptitle('Grad-CAM: Model Attention on Different Dementia Stages', fontsize=16)

    for i, (class_idx, image_tensor) in enumerate(images_per_class.items()):
        if image_tensor is None:
            print(f"Warning: Could not find an image for class {class_names[class_idx]} in the first batches.")
            continue

        input_tensor = image_tensor.unsqueeze(0).to(device)
        # Normalize the image for display purposes
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


# ================================================================================
# SECTION 1.5: CUSTOM LOSS FUNCTION DEFINITION
# ================================================================================

class BalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2.0, class_thresholds=None):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)
        self.class_thresholds = class_thresholds if class_thresholds is not None else {1: 5.0}

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        class_weights_applied = self.class_weights[targets]

        extra_weights = torch.ones_like(targets, dtype=torch.float32)
        for class_idx, multiplier in self.class_thresholds.items():
            extra_weights[targets == class_idx] *= multiplier

        final_weights = focal_weight * class_weights_applied * extra_weights.to(inputs.device)
        return (final_weights * ce_loss).mean()


def create_weighted_focal_loss(class_weights, alpha=0.25, gamma=2.0):
    # This helper function instantiates your CBFL with specific parameters
    gamma = 3.0  # As defined in your ViT script
    class_frequencies = class_weights / class_weights.sum()
    class_thresholds = {
        i: 8.0 if freq < 0.01 else (5.0 if freq < 0.05 else 1.0)
        for i, freq in enumerate(class_frequencies)
    }
    return BalancedFocalLoss(
        class_weights=class_weights,
        alpha=alpha,
        gamma=gamma,
        class_thresholds=class_thresholds
    )


# ================================================================================
# REVISED EVALUATION FUNCTION
# This version corrects the bug and is structured for robust reporting.
# ================================================================================
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, data_loader, class_names, device, set_name=""):
    """
    Evaluates the model on a given dataset and returns key performance metrics.
    Generates and saves a confusion matrix plot.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- METRICS CALCULATION AND REPORTING ---

    # Ensure the lists are numpy arrays for scikit-learn
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 1. Calculate overall accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\nOverall Test Accuracy on '{set_name}' set: {accuracy:.2f}%")

    # 2. Generate and print the classification report
    # Using labels=range(len(class_names)) ensures all classes appear in the report,
    # even if one is missing from the test set, preventing errors.
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0,
        output_dict=True  # Return the report as a dictionary
    )
    # Convert dictionary to a printable string format
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        labels=range(len(class_names)),
        zero_division=0
    )
    print(report_str)

    # 3. Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix ({set_name} set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='center')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{set_name}.png')
    plt.close()
    print(f"Saved confusion matrix to confusion_matrix_{set_name}.png")

    # Return all key metrics for logging or further analysis
    return accuracy, cm, report


if __name__ == '__main__':
    # --- Configuration (Hyperparameters are unchanged as requested) ---
    DATA_DIR = r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data"
    CLASSES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
    N_SPLITS = 5
    TEST_SET_RATIO = 0.2
    RANDOM_STATE = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # --- Data Preparation ---
    all_subjects, all_labels, subjects_to_images, subject_to_label = prepare_subject_data(DATA_DIR, CLASSES)
    dev_subjects, test_subjects, dev_labels, _ = train_test_split(
        all_subjects, all_labels, test_size=TEST_SET_RATIO, random_state=RANDOM_STATE, stratify=all_labels
    )

    # --- K-Fold Cross-Validation ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results, best_fold_acc, best_model_state = [], 0.0, None

    print(f"\nStarting {N_SPLITS}-Fold Cross-Validation for EfficientNet-B0 with Standard CE Loss...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_subjects, dev_labels)):
        print(f"\n===== FOLD {fold + 1}/{N_SPLITS} =====")
        train_fold_subjects, val_fold_subjects = [dev_subjects[i] for i in train_idx], [dev_subjects[i] for i in
                                                                                        val_idx]
        train_images, train_labels = unpack_split_data(train_fold_subjects, subjects_to_images, subject_to_label)
        val_images, val_labels = unpack_split_data(val_fold_subjects, subjects_to_images, subject_to_label)

        train_dataset = DementiaDataset(train_images, train_labels, transform=train_transform)
        val_dataset = DementiaDataset(val_images, val_labels, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        # Re-initialize model and optimizer for each fold for a fair comparison
        # Re-initialize model and optimizer for each fold
        model = DementiaModel(num_classes=len(CLASSES)).to(device)

        # --- ADD THIS BLOCK: Calculate weights for the current fold's training data ---
        class_counts = torch.tensor([train_labels.count(i) for i in range(len(CLASSES))]).float()
        class_weights = 1.0 / (class_counts / class_counts.sum())

        # --- REPLACE THE CRITERION LINE WITH THIS ---
        criterion = create_weighted_focal_loss(class_weights=class_weights.to(device))

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3, amsgrad=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

        trained_model, fold_accuracy = train_and_validate_fold(model, train_loader, val_loader, criterion, optimizer,
                                                               scheduler, device, NUM_EPOCHS)
        fold_results.append(fold_accuracy)

        if fold_accuracy > best_fold_acc:
            best_fold_acc = fold_accuracy
            best_model_state = copy.deepcopy(trained_model.state_dict())

    # --- Post-Loop Summary ---
    print(f"\n\n===== K-FOLD CROSS-VALIDATION SUMMARY =====")
    print(
        f"Average Validation Accuracy across {N_SPLITS} folds: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})\n")

    # --- Final Evaluation and Visualization on the Held-Out Test Set ---
    print("\n===== FINAL EVALUATION ON HELD-OUT TEST SET =====")
    test_images, test_labels = unpack_split_data(test_subjects, subjects_to_images, subject_to_label)
    test_dataset = DementiaDataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load the best model found during cross-validation
    final_model = DementiaModel(num_classes=len(CLASSES)).to(device)
    final_model.load_state_dict(best_model_state)

    # 1. Call your original, verbose evaluation function
    # This will print the classification report and save the confusion matrix plot
    test_accuracy, confusion_mat = evaluate_model(final_model, test_loader, CLASSES, set="test")
    print(f"\nFinal Test Set Accuracy: {test_accuracy:.4f}")
    print("\nFinal Confusion Matrix:\n", confusion_mat)

    # 2. Call the ROC/PR curve generation function
    generate_roc_pr_curves(final_model, test_loader, device, CLASSES, file_prefix="efficientnetb0_final")

    # 3. Call the Grad-CAM generation function
    target_layer = [final_model.model.features[-1]]
    generate_grad_cam_visualizations(final_model, test_loader, device, target_layer, CLASSES,
                                     file_prefix="efficientnetb0_final")

    print("\n--- All Processes Complete ---")