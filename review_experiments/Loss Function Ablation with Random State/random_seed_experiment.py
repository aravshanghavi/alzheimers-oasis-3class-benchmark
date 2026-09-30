# ================================================================================
# SECTION 1: CONSOLIDATED IMPORTS
# ================================================================================

# --- Standard Library ---
import os
import re
import warnings
from collections import defaultdict, Counter
from itertools import cycle

# --- Core Libraries ---
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- Machine Learning & Deep Learning Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from torchvision import models, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# --- Visualization & Analysis Tools ---
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Global Settings ---
warnings.filterwarnings('ignore')

# ================================================================================
# SECTION 1: CONFIGURATION
# Centralized parameters for the data preparation pipeline.
# ================================================================================
CONFIG = {
    "DATA_DIR": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data",
    "RANDOM_STATE": 256,
    "TEST_SET_RATIO": 0.20,
    "VAL_SET_RATIO": 0.15,
    "BATCH_SIZE": 32,
    "ORIGINAL_CLASSES": ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"],
    "NEW_CLASSES": ["Non Demented", "Very Mild Demented", "Demented"],
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "NUM_EPOCHS": 20,
    "LEARNING_RATE": 1e-4,

    # --- NEW: Experiment Configuration Block ---
    "LOSS": {
        # Options: "WCE", "FA_FL", "FocalLoss", "LDAM"
        "TYPE": "FA_FL",

        # Parameters for each loss type
        "PARAMS": {
            "FA_FL": {"gamma_base": 1.0, "lambda_val": 7.0},
            "FocalLoss": {"gamma": 3.0},
            "LDAM": {"s": 30.0} # This is the scaling factor C from the paper
        }
    }
}

# ================================================================================
# SECTION 2: DATASET CLASS & INTERNAL DATA HELPERS
# ================================================================================
class DementiaDataset(Dataset):
    """Standard Dataset class for loading images."""
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
        return image, torch.tensor(label, dtype=torch.long)


def _prepare_subject_data(data_dir, classes):
    """(Internal) Scans directories and groups all images by subject ID."""
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
    return list(subjects_to_images.keys()), subjects_to_images, subject_to_label


def _unpack_split_data(subject_list, subjects_to_images, subject_to_label):
    """(Internal) Expands a list of subject IDs into corresponding image paths and labels."""
    images, labels = [], []
    for subj in subject_list:
        subj_images = subjects_to_images[subj]
        images.extend(subj_images)
        labels.extend([subject_to_label[subj]] * len(subj_images))
    return images, labels


def _calculate_class_weights(train_img_labels, class_names, device):
    """(Internal) Calculates normalized inverse frequency class weights for the training set."""
    class_counts = Counter(train_img_labels)
    sorted_counts = [class_counts[i] for i in range(len(class_names))]
    total_samples = sum(sorted_counts)
    weights = [total_samples / count if count > 0 else 0 for count in sorted_counts]
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]
    return torch.tensor(normalized_weights, dtype=torch.float32).to(device)

# ADD THIS HELPER FUNCTION TO SECTION 2

def _get_class_counts(train_img_labels, class_names):
    """(Internal) Calculates the number of training samples per class."""
    class_counts = Counter(train_img_labels)
    # Ensure the counts are in the correct order (0, 1, 2...)
    sorted_counts = [class_counts[i] for i in range(len(class_names))]
    return sorted_counts


# ================================================================================
# SECTION 3: VISUALIZATION & REPORTING FUNCTIONS
# ================================================================================

# NEW: Helper function to create experiment-specific output directories
def get_and_create_output_dir(config):
    """Creates and returns the path to a directory for saving experiment artifacts."""
    loss_type = config["LOSS"]["TYPE"]
    seed = config["RANDOM_STATE"]
    output_dir = os.path.join(loss_type, str(seed))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def visualize_preprocessing_steps(image_path, output_dir):
    """Visualizes and saves each step of the preprocessing pipeline for a single image."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Steps Visualization', fontsize=16)
    original_img = Image.open(image_path).convert('RGB')
    axes[0, 0].imshow(original_img); axes[0, 0].set_title('Original Image')
    resized_img = transforms.Resize((224, 224))(original_img)
    axes[0, 1].imshow(resized_img); axes[0, 1].set_title('Resized (224x224)')
    axes[0, 2].imshow(transforms.RandomHorizontalFlip(p=1)(resized_img)); axes[0, 2].set_title('Horizontal Flip')
    axes[1, 0].imshow(transforms.RandomVerticalFlip(p=1)(resized_img)); axes[1, 0].set_title('Vertical Flip')
    normalized_img_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(transforms.ToTensor()(resized_img))
    # Rescale tensor from [-1, 1] to [0, 1] for visualization
    display_tensor = (normalized_img_tensor - normalized_img_tensor.min()) / (normalized_img_tensor.max() - normalized_img_tensor.min())
    axes[1, 1].imshow(display_tensor.permute(1, 2, 0)); axes[1, 1].set_title('Normalized')
    final_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    final_img_tensor = final_transform(original_img)
    display_tensor_final = (final_img_tensor - final_img_tensor.min()) / (final_img_tensor.max() - final_img_tensor.min())
    axes[1, 2].imshow(display_tensor_final.permute(1, 2, 0)); axes[1, 2].set_title('Example Final Transform')
    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, 'preprocessing_steps.png')) ; plt.close()


def visualize_class_distribution(train_labels, classes, class_weights, output_dir):
    """Visualizes and saves the class distribution before and after applying loss weighting."""
    class_counts = [train_labels.count(idx) for idx in range(len(classes))]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    fig.suptitle('Training Set Class Distribution', fontsize=16)
    x = np.arange(len(classes))
    ax1.bar(x, class_counts, color='skyblue')
    ax1.set_title('Original Sample Count'); ax1.set_ylabel('Number of Images')
    ax1.set_xticks(x); ax1.set_xticklabels(classes, rotation=45, ha='right')
    # This shows the total "importance" of each class after weighting
    weighted_importance = [count * class_weights[i].item() for i, count in enumerate(class_counts)]
    ax2.bar(x, weighted_importance, color='salmon')
    ax2.set_title('Effective Importance After Weighting'); ax2.set_ylabel('Cumulative Loss Contribution')
    ax2.set_xticks(x); ax2.set_xticklabels(classes, rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, 'class_distribution.png')); plt.close()


def create_dataset_summary(train_images, val_images, test_images, output_dir):
    """Creates and saves a visualization summarizing the dataset splits."""
    sizes = [len(train_images), len(val_images), len(test_images)]
    total_size = sum(sizes)
    splits = ['Train', 'Validation', 'Test']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.5]})
    fig.suptitle('Dataset Split Summary (by Image Count)', fontsize=16)
    ax1.pie(sizes, labels=splits, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax1.set_title('Data Split by Percentage')
    table_data = [[f"{size:,}"] for size in sizes]
    table_data.append([f"{total_size:,}"])
    row_labels = splits + ['Total']
    table = ax2.table(cellText=table_data, rowLabels=row_labels, colLabels=['Image Count'], loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 2)
    ax2.axis('off'); ax2.set_title('Data Split by Absolute Count')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, 'dataset_summary.png')); plt.close()

# ================================================================================
# SECTION 4: PUBLIC API FUNCTION (DATA PREPARATION)
# ================================================================================
def get_data_products(config, device):
    """
    Orchestrates the data loading pipeline and returns a dictionary of all data products.
    This is the sole entry point for the data preparation module.
    """
    print("===== EXECUTING DATA PREPARATION PIPELINE =====")
    # Step 1: Load raw data and group by subject
    all_subjects, subjects_to_images, subject_to_original_label = _prepare_subject_data(config["DATA_DIR"], config["ORIGINAL_CLASSES"])

    # Step 2: Remap subjects to the new 3-class problem
    class_mapping = {
        config["ORIGINAL_CLASSES"].index("Non Demented"): config["NEW_CLASSES"].index("Non Demented"),
        config["ORIGINAL_CLASSES"].index("Very mild Dementia"): config["NEW_CLASSES"].index("Very Mild Demented"),
        config["ORIGINAL_CLASSES"].index("Mild Dementia"): config["NEW_CLASSES"].index("Demented"),
        config["ORIGINAL_CLASSES"].index("Moderate Dementia"): config["NEW_CLASSES"].index("Demented")
    }
    subject_to_new_label = {s: class_mapping[li] for s, li in subject_to_original_label.items()}
    all_new_labels = [subject_to_new_label[s] for s in all_subjects]

    # Step 3: Perform strict, subject-level train/val/test split
    dev_subjects, test_subjects, dev_labels, _ = train_test_split(all_subjects, all_new_labels, test_size=config["TEST_SET_RATIO"], random_state=config["RANDOM_STATE"], stratify=all_new_labels)
    val_split_ratio = config["VAL_SET_RATIO"] / (1 - config["TEST_SET_RATIO"])
    train_subjects, val_subjects, _, _ = train_test_split(dev_subjects, dev_labels, test_size=val_split_ratio, random_state=config["RANDOM_STATE"], stratify=dev_labels)

    # Step 4: Unpack splits into final image lists and create DataLoaders
    train_images, train_labels = _unpack_split_data(train_subjects, subjects_to_images, subject_to_new_label)
    val_images, val_labels = _unpack_split_data(val_subjects, subjects_to_images, subject_to_new_label)
    test_images, test_labels = _unpack_split_data(test_subjects, subjects_to_images, subject_to_new_label)

    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    val_test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_dataset = DementiaDataset(train_images, train_labels, transform=train_transform)
    val_dataset = DementiaDataset(val_images, val_labels, transform=val_test_transform)
    test_dataset = DementiaDataset(test_images, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)

    # Step 5: Calculate class weights and counts from the training set
    class_weights = _calculate_class_weights(train_labels, config["NEW_CLASSES"], device)
    class_counts = _get_class_counts(train_labels, config["NEW_CLASSES"])  # Add this line

    return {
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "class_weights": class_weights,
        "class_counts": class_counts,  # Add this new key
        "image_lists": {"train": train_images, "val": val_images, "test": test_images},
        "labels_lists": {"train": train_labels, "val": val_labels, "test": test_labels},
        "class_names": config["NEW_CLASSES"]
    }


# ================================================================================
# SECTION 5: MODEL, LOSS, AND TRAINING LOGIC
# This module contains the core components for the experiment.
# ================================================================================

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss.
    This loss function enforces a class-dependent margin on the logits, with
    larger margins for minority classes, to improve performance on imbalanced datasets.
    Reference: https://arxiv.org/abs/1906.07413
    """

    def __init__(self, cls_num_list, s=30.0):
        """
        Args:
            cls_num_list (list): A list containing the number of samples for each class.
            s (float): A scaling factor, typically 30.
        """
        super(LDAMLoss, self).__init__()
        # Calculate the class-specific margins. The margin is inversely proportional to the 4th root of the class size.
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # Scale the margins to prevent them from being too small.
        m_list = m_list * (max(m_list) / min(m_list))
        m_list = torch.FloatTensor(m_list)
        # Register as a buffer so it moves to the correct device with the model.
        self.register_buffer('m_list', m_list)
        self.s = s  # This is the scaling factor s from the paper.

    def forward(self, inputs, targets):
        # Create a one-hot-like tensor for indexing the correct class logits.
        index = torch.zeros_like(inputs, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)

        # Create a batch-specific margin tensor.
        # --- THIS IS THE CORRECTED LINE ---
        # We move m_list to the same device as the inputs/index tensor before the multiplication.
        batch_m = torch.matmul(self.m_list[None, :].to(inputs.device), index.transpose(0, 1).float())
        batch_m = batch_m.view((-1, 1))

        # Subtract the margin ONLY from the logits of the correct class.
        x_m = inputs - batch_m

        # Use torch.where to combine the modified logits with the original ones.
        output = torch.where(index, x_m, inputs)

        # Return the final cross-entropy loss, scaled by s.
        return F.cross_entropy(self.s * output, targets)

class StandardFocalLoss(nn.Module):
    """
    Standard Focal Loss implementation.
    This loss function applies a modulating factor to the cross-entropy loss
    to focus training on hard-to-classify examples. It uses a fixed gamma
    and pre-calculated alpha (class weights).
    """

    def __init__(self, alpha, gamma=2.0):
        """
        Args:
            alpha (torch.Tensor): A tensor of class weights. Shape (num_classes,).
            gamma (float): The fixed focusing parameter.
        """
        super(StandardFocalLoss, self).__init__()
        if not isinstance(alpha, torch.Tensor):
            raise TypeError("alpha must be a torch.Tensor")
        # Use register_buffer for non-parameter tensors that should move to the device with the model
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate the cross-entropy loss, but don't reduce it to a single value yet
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate pt, the model's estimated probability of the true class
        pt = torch.exp(-ce_loss)

        # Gather the alpha weight for each sample in the batch based on its target class
        alpha_t = self.alpha.gather(0, targets)

        # Calculate the final focal loss
        focal_loss = alpha_t * torch.pow(1 - pt, self.gamma) * ce_loss

        # Return the mean of the loss over the batch
        return focal_loss.mean()

class FrequencyAdaptiveFocalLoss(nn.Module):
    """
    Frequency-Adaptive Focal Loss (FA-FL). This loss function adapts the
    focusing parameter `gamma` for each class based on its rarity.
    """

    def __init__(self, alpha, gamma_base=2.0, lambda_val=1.0):
        super(FrequencyAdaptiveFocalLoss, self).__init__()
        if not isinstance(alpha, torch.Tensor):
            raise TypeError("alpha must be a torch.Tensor")
        self.register_buffer('alpha', alpha)
        gammas = gamma_base + lambda_val * self.alpha
        self.register_buffer('gammas', gammas)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha.gather(0, targets)
        gamma_t = self.gammas.gather(0, targets)
        modulating_factor = alpha_t * torch.pow(1 - pt, gamma_t)
        focal_loss = modulating_factor * ce_loss
        return focal_loss.mean()

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


class DementiaModel(nn.Module):
    """
    The deep learning model architecture, with a simplified and heavily regularized head.
    """
    def __init__(self, num_classes):
        super(DementiaModel, self).__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        # The entire backbone is now one unit
        self.model.classifier = nn.Identity()
        # The classifier head is a separate unit
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.7),  # Increased dropout for aggressive regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features)
        return output


class EarlyStopping:
    """Helper class to stop training when validation loss stops improving."""

    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')  # More robust initialization
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose: print("--- Early stopping triggered ---")
                self.early_stop = True


def train_model(model, train_loader, val_loader, criterion, config, output_dir):
    """
    Trains the model using the corrected two-stage protocol with AMP and Gradient Clipping.
    """
    device = config["DEVICE"]
    model.to(device)
    save_model_path = os.path.join(output_dir, "best_model_weights.pth")

    # --- STAGE 1: WARM-UP / FEATURE EXTRACTION ---
    # Freeze all layers in the backbone and define an optimizer for the head only
    for param in model.model.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.classifier.parameters(), lr=config["LEARNING_RATE"])

    num_warmup_epochs = 5
    print(f"\n===== STARTING STAGE 1: WARM-UP (Training Classifier Head for {num_warmup_epochs} epochs) =====")
    for epoch in range(num_warmup_epochs):
        print(f"\n--- Warm-up Epoch {epoch + 1}/{num_warmup_epochs} ---")
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # --- STAGE 2: FINE-TUNING ---
    # Unfreeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Re-initialize optimizer for the FULL model with a very low learning rate
    fine_tune_lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=1e-4)

    num_finetune_epochs = config.get("NUM_EPOCHS", 20) - num_warmup_epochs
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_finetune_epochs, eta_min=1e-7)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    accumulation_steps = 2

    print(f"\n===== STARTING STAGE 2: FINE-TUNING (Training Full Model with LR={fine_tune_lr}) =====")
    for epoch in range(num_finetune_epochs):
        print(f"\n--- Fine-tuning Epoch {epoch + 1}/{num_finetune_epochs} ---")

        # --- Training Phase with AMP, Gradient Accumulation, and Clipping ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        optimizer.zero_grad()  # Zero gradients once at the start of the accumulation

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Normalize loss for accumulation

            scaler.scale(loss).backward()

            # --- THE ENTIRE UPDATE BLOCK IS MOVED INSIDE THE IF STATEMENT ---
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero gradients after the step
            # ----------------------------------------------------------------

            _, preds = torch.max(outputs, 1)
            # We multiply by accumulation_steps to get the "true" loss for logging
            running_loss += loss.item() * images.size(0) * accumulation_steps
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)

        # --- Validation Phase (Streamlined) ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = (val_corrects.double() / len(val_loader.dataset)) * 100

        val_accuracy, val_cm = evaluate_model(
            model=model,
            data_loader=data['loaders']['val'],
            class_names=data['class_names'],
            set_name="val",
            output_dir=output_dir
        )

        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        print(f"\n{val_cm}\n")

        if val_epoch_acc > best_acc:
            print(f"  Validation accuracy improved ({best_acc:.2f}% --> {val_epoch_acc:.2f}%). Saving model...")
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), save_model_path)

        scheduler.step()
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    # Load the best performing model weights
    if os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path))
    return model

def evaluate_model(model, data_loader, class_names, set_name, output_dir):
    """
    Evaluates the final model on a given dataset, saves a confusion matrix, and returns accuracy and the matrix.
    """
    print(f"\n===== EVALUATING MODEL ON {set_name.upper()} SET =====")
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nClassification Report ({set_name} set):")
    # Using labels=range(...) ensures all classes appear in the report, even if not predicted
    print(classification_report(all_labels, all_preds, target_names=class_names, labels=range(len(class_names)),
                                zero_division=0))

    # Calculate overall accuracy manually for a clear printout
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Overall Accuracy ({set_name} set): {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({set_name} Set)')
    plt.xlabel('Predicted Label');
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{set_name}.png'))
    plt.close()
    print(f"-> Saved '{f'confusion_matrix_{set_name}.png'}'")

    return accuracy, cm


# ================================================================================
# SECTION 6: POST-TRAINING VISUALIZATION FUNCTIONS
# ================================================================================

def generate_roc_pr_curves(model, test_loader, device, class_names, output_dir):
    """
    Generates and saves ROC and Precision-Recall curve plots for the test set.
    """
    print("\n--- Generating ROC and Precision-Recall curves ---")
    model.eval()
    y_true, y_probas = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probas = F.softmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_probas.extend(probas)

    y_true = np.array(y_true)
    y_probas = np.array(y_probas)
    n_classes = len(class_names)

    # --- ROC Curve Generation ---
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        y_true_class = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['cornflowerblue', 'darkorange', 'forestgreen', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve for {class_names[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    # --- Precision-Recall Curve Generation ---
    precision, recall, avg_precision = dict(), dict(), dict()
    for i in range(n_classes):
        y_true_class = (y_true == i).astype(int)
        precision[i], recall[i], _ = precision_recall_curve(y_true_class, y_probas[:, i])
        avg_precision[i] = average_precision_score(y_true_class, y_probas[:, i])

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve for {class_names[i]} (AP = {avg_precision[i]:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves per Class')
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, "pr_curves.png"))
    plt.close()


def generate_grad_cam_visualizations(model, test_loader, device, target_layers, class_names, file_name, output_dir):
    """
    Generates and saves Grad-CAM visualizations for one sample from each class.
    """
    print("\n--- Generating Grad-CAM visualizations ---")
    model.eval()
    num_classes = len(class_names)
    images_per_class = {i: None for i in range(num_classes)}

    # Find one image for each class from the test set
    for images, labels in test_loader:
        for i in range(len(labels)):
            label = labels[i].item()
            if images_per_class[label] is None:
                images_per_class[label] = images[i]
        if all(v is not None for v in images_per_class.values()):
            break

    cam = GradCAM(model=model, target_layers=target_layers)

    # Determine subplot layout
    rows = int(np.ceil(num_classes / 3))
    cols = min(num_classes, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    fig.suptitle('Grad-CAM: Model Attention on Different Dementia Stages', fontsize=16)

    for i, (class_idx, image_tensor) in enumerate(images_per_class.items()):
        ax = axes[i]
        if image_tensor is None:
            ax.set_title(f"No image found for\n{class_names[class_idx]}")
            ax.axis('off');
            continue

        input_tensor = image_tensor.unsqueeze(0).to(device)
        rgb_img = image_tensor.permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))

        targets = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax.imshow(visualization)
        ax.set_title(f"True Class: {class_names[class_idx]}")
        ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    file_name = os.path.join(output_dir, file_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':
    # Get and create the unique output directory for this specific run
    output_dir = get_and_create_output_dir(CONFIG)

    print(f"===== STARTING RUN | SEED: {CONFIG['RANDOM_STATE']} | LOSS: {CONFIG['LOSS']['TYPE']} | PARAMS: {CONFIG['LOSS']['PARAMS'][CONFIG['LOSS']['TYPE']]}  =====")
    print(f"Output will be saved to: {output_dir}")

    # --- Step 1: Run data preparation ---
    data = get_data_products(CONFIG, CONFIG['DEVICE'])
    print("\n===== DATA PREPARATION SUMMARY =====")
    print(f"  Class Names:  {data['class_names']}")
    print(f"  Class Weights: {data['class_weights'].cpu().numpy()}")

    # --- Step 2: Generate and save dataset visualizations ---
    print("\n===== GENERATING DATASET VISUALIZATIONS =====")
    if data['image_lists']['train']:
        visualize_preprocessing_steps(data['image_lists']['train'][0], output_dir)
    visualize_class_distribution(data['labels_lists']['train'], data['class_names'], data['class_weights'], output_dir)
    create_dataset_summary(data['image_lists']['train'], data['image_lists']['val'], data['image_lists']['test'],
                           output_dir)

    # --- Step 3: Initialize Model and Loss ---
    print("\n===== INITIALIZING MODEL AND CRITERION =====")
    model = DementiaModel(num_classes=len(data['class_names']))

    # (Your dynamic criterion block - this is correct)
    loss_type = CONFIG["LOSS"]["TYPE"]
    loss_params = CONFIG["LOSS"]["PARAMS"]
    if loss_type == "WCE":
        criterion = nn.CrossEntropyLoss(weight=data['class_weights'])
    elif loss_type == "FA_FL" or loss_type == "FA_FL2":
        params = loss_params["FA_FL"]
        criterion = FrequencyAdaptiveFocalLoss(alpha=data['class_weights'], **params)
    elif loss_type == "FocalLoss":
        params = loss_params["FocalLoss"]
        criterion = StandardFocalLoss(alpha=data['class_weights'], **params)
    elif loss_type == "LDAM":
        criterion = LDAMLoss(cls_num_list=data.get('class_counts', []), **loss_params["LDAM"])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # --- Step 4: Train the Model ---
    trained_model = train_model(
        model=model,
        train_loader=data['loaders']['train'],
        val_loader=data['loaders']['val'],
        criterion=criterion,
        config=CONFIG,
        output_dir=output_dir
    )

    # --- Step 5: Final Evaluation and Visualizations ---
    print("\n===== FINAL PERFORMANCE EVALUATION ON TEST SET =====")
    test_accuracy, test_cm = evaluate_model(
        model=trained_model,
        data_loader=data['loaders']['test'],
        class_names=data['class_names'],
        set_name="test",
        output_dir=output_dir
    )

    generate_roc_pr_curves(
        model=trained_model,
        test_loader=data['loaders']['test'],
        device=CONFIG['DEVICE'],
        class_names=data['class_names'],
        output_dir=output_dir
    )

    target_layer = [trained_model.model.features[-1]]
    generate_grad_cam_visualizations(
        model=trained_model,
        test_loader=data['loaders']['test'],
        device=CONFIG['DEVICE'],
        target_layers=target_layer,
        class_names=data['class_names'],
        file_name="grad_cam_final_results.png",
        output_dir=output_dir
    )

    print("\n--- Run Complete ---")