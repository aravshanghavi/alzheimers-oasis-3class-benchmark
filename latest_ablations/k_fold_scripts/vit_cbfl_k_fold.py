# ================================================================================
# DEFINITIVE K-FOLD CROSS-VALIDATION SCRIPT FOR ViT + CBFL
# This script implements the complete, robust evaluation pipeline required for
# a Tier 1 publication, resolving all data splitting issues.
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
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, \
    classification_report

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
# SECTION 1: MODEL AND LOSS FUNCTION DEFINITIONS
# ================================================================================

class DementiaModel(nn.Module):
    def __init__(self, num_classes):
        super(DementiaModel, self).__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = models.vit_b_16(weights=weights)
        self.model.heads = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


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
    gamma = 3.0
    total_samples = class_weights.sum()
    class_frequencies = class_weights / total_samples
    class_thresholds = {i: 8.0 if freq < 0.01 else (5.0 if freq < 0.05 else 1.0) for i, freq in
                        enumerate(class_frequencies)}
    return BalancedFocalLoss(class_weights=class_weights, alpha=alpha, gamma=gamma, class_thresholds=class_thresholds)


# ================================================================================
# SECTION 2: DATASET AND PREPARATION FUNCTIONS
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


# ================================================================================
# SECTION 3: CORE TRAINING AND EVALUATION LOGIC
# ================================================================================

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
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

        val_acc, _ = evaluate_fold(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch + 1}/{num_epochs} -> Val Acc: {val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, best_acc


def evaluate_fold(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return accuracy, f1


def get_predictions_and_labels(model, data_loader, device):
    model.eval()
    y_true, y_probas = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probas = F.softmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_probas.extend(probas)
    return np.array(y_true), np.array(y_probas)


# ================================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ================================================================================

def generate_roc_pr_curves(y_true, y_probas, class_names, file_prefix=""):
    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))
    # ROC
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        y_true_class = np.array(y_true) == i
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
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
    print(f"Saved ROC curves to {file_prefix}_roc_curves.png")

    # PR
    precision, recall, avg_precision = dict(), dict(), dict()
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        y_true_class = np.array(y_true) == i
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
    print(f"Saved PR curves to {file_prefix}_pr_curves.png")


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def generate_grad_cam_visualizations(model, data_loader, device, target_layers, class_names, file_prefix="",
                                     reshape_transform=None):
    images_per_class = {i: None for i in range(len(class_names))}
    for images, labels in data_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if images_per_class.get(label) is None: images_per_class[label] = images[i]
        if all(v is not None for v in images_per_class.values()): break
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, (class_idx, image_tensor) in enumerate(images_per_class.items()):
        ax = axes.flat[i]
        if image_tensor is None:
            ax.set_title(f"No image for class '{class_names[class_idx]}'");
            ax.axis('off');
            continue
        input_tensor = image_tensor.unsqueeze(0).to(device)
        rgb_img = image_tensor.permute(1, 2, 0).numpy();
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        ax.imshow(visualization);
        ax.set_title(f"True Class: {class_names[class_idx]}");
        ax.axis('off')
    plt.savefig(f"{file_prefix}_gradcam.png");
    plt.close()
    print(f"Saved Grad-CAM image to {file_prefix}_gradcam.png")


# ================================================================================
# SECTION 5: MAIN K-FOLD EXECUTION BLOCK
# ================================================================================
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data"
    CLASSES = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
    N_SPLITS = 5
    TEST_SET_RATIO = 0.2
    RANDOM_STATE = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Transforms ---
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # --- Data Preparation ---
    all_subjects, all_labels, subjects_to_images, subject_to_label = prepare_subject_data(DATA_DIR, CLASSES)
    dev_subjects, test_subjects, dev_labels, _ = train_test_split(all_subjects, all_labels, test_size=TEST_SET_RATIO,
                                                                  random_state=RANDOM_STATE, stratify=all_labels)

    # --- K-Fold Cross-Validation ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results, best_fold_acc, best_model_state = [], 0.0, None

    print(f"\nStarting {N_SPLITS}-Fold Cross-Validation for ViT with CBFL...")
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

        model = DementiaModel(num_classes=len(CLASSES)).to(device)
        class_counts = torch.tensor([train_labels.count(i) for i in range(len(CLASSES))]).float()
        class_weights = 1.0 / (class_counts / class_counts.sum())

        criterion = create_weighted_focal_loss(class_weights=class_weights.to(device))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3, amsgrad=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

        trained_model, fold_accuracy = train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                  device, NUM_EPOCHS)
        fold_results.append(fold_accuracy)

        if fold_accuracy > best_fold_acc:
            best_fold_acc = fold_accuracy
            best_model_state = copy.deepcopy(trained_model.state_dict())

    print(
        f"\n\n===== K-FOLD CROSS-VALIDATION SUMMARY =====\nAverage Validation Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})\n")

    # --- Final Evaluation and Visualization ---
    print("\n===== FINAL EVALUATION ON HELD-OUT TEST SET =====")
    test_images, test_labels = unpack_split_data(test_subjects, subjects_to_images, subject_to_label)
    test_dataset = DementiaDataset(test_images, test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    final_model = DementiaModel(num_classes=len(CLASSES)).to(device)
    final_model.load_state_dict(best_model_state)

    y_true_test, y_probas_test = get_predictions_and_labels(final_model, test_loader, device)

    print("\nFinal Test Set Classification Report:")
    print(classification_report(y_true_test, y_probas_test.argmax(axis=1), target_names=CLASSES, zero_division=0))

    print("\n--- Generating Final Visualizations ---")
    generate_roc_pr_curves(y_true_test, y_probas_test, CLASSES, file_prefix="vit_cbfl_final")

    target_layer_vit = [final_model.model.encoder.layers[-1].ln_1]
    generate_grad_cam_visualizations(final_model, test_loader, device, target_layer_vit, CLASSES,
                                     file_prefix="vit_cbfl_final", reshape_transform=reshape_transform_vit)

    print("\n--- All Processes Complete ---")