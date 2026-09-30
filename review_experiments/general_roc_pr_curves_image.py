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
                             precision_recall_curve, average_precision_score, roc_auc_score)
from torchvision import models, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# --- Visualization & Analysis Tools ---
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Global Settings ---
warnings.filterwarnings('ignore')

# ================================================================================
# SECTION 1: CONFIGURATION
# ================================================================================
CONFIG = {
    # --- NEW: Execution Mode ('train' or 'evaluate') ---
    "MODE": "evaluate",  # Set to 'evaluate' to skip training and use pre-trained weights

    "DATA_DIR": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data",
    "SEEDS": [42, 88, 101, 256, 2024],
    "TEST_SET_RATIO": 0.20,
    "VAL_SET_RATIO": 0.15,
    "BATCH_SIZE": 32,
    "ORIGINAL_CLASSES": ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"],
    "NEW_CLASSES": ["Non Demented", "Very Mild Demented", "Demented"],
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # --- Training Block (used only if MODE is 'train') ---
    "TRAINING": {
        "NUM_EPOCHS": 20,
        "WARMUP_EPOCHS": 5,
        "LEARNING_RATE_WARMUP": 1e-4,
        "LEARNING_RATE_FINETUNE": 1e-5,
        "WEIGHT_DECAY": 1e-4,
        "EARLY_STOPPING_PATIENCE": 5,
        "GRAD_ACCUMULATION_STEPS": 2,
        "GRAD_MAX_NORM": 1.0
    },

    # --- Loss Configuration (used only if MODE is 'train') ---
    "LOSS": {
        "TYPE": "FA_FL",
        "PARAMS": {
            "FA_FL": {"gamma_base": 1.0, "lambda_val": 7.0},
            "FocalLoss": {"gamma": 3.0},
            "LDAM": {"s": 30.0}
        }
    }
}

# ================================================================================
# SECTION 2: WEIGHTS & EVALUATION SETUP (ONLY FOR 'evaluate' MODE)
# ================================================================================

# ✅ ACTION REQUIRED: FILL THIS DICTIONARY WITH YOUR WEIGHT PATHS
# Structure: WEIGHT_PATHS['LOSS_TYPE'][SEED] = 'path/to/your/weights.pth'
WEIGHT_PATHS = {
    "WCE": {
        42: r"C:\Users\aravs\PycharmProjects\Alzheimers\new_strategy\Random Seed\new_wce\42\best_model_weights.pth",
        88: r"C:\Users\aravs\PycharmProjects\Alzheimers\new_strategy\Random Seed\new_wce\88\best_model_weights.pth",
        101: r"C:\Users\aravs\PycharmProjects\Alzheimers\new_strategy\Random Seed\new_wce\101\best_model_weights.pth",
        256: r"C:\Users\aravs\PycharmProjects\Alzheimers\new_strategy\Random Seed\new_wce\256\best_model_weights.pth",
        2024: r"C:\Users\aravs\PycharmProjects\Alzheimers\new_strategy\Random Seed\new_wce\2024\best_model_weights.pth",
    },
    "LDAM": {
        42: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\LDAM\42\best_model_weights.pth",
        88: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\LDAM\88\best_model_weights.pth",
        101: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\LDAM\101\best_model_weights.pth",
        256: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\LDAM\256\best_model_weights.pth",
        2024: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\LDAM\2024\best_model_weights.pth",
    },
    "FocalLoss": {
        42: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FocalLoss\42\best_model_weights.pth",
        88: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FocalLoss\88\best_model_weights.pth",
        101: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FocalLoss\101\best_model_weights.pth",
        256: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FocalLoss\256\best_model_weights.pth",
        2024: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FocalLoss\2024\best_model_weights.pth",
    },
    "FA_FL": {
        42: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FA_FL\42\best_model_weights.pth",
        88: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FA_FL\88\best_model_weights.pth",
        101: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FA_FL\101\best_model_weights.pth",
        256: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FA_FL\256\best_model_weights.pth",
        2024: r"C:\Users\aravs\PycharmProjects\Alzheimers\review_experiments\Loss Function Ablation with Random State\FA_FL\2024\best_model_weights.pth",
    }
}

# This map defines which seed to use for visualization for each model type.
# Based on our analysis of finding the run closest to the mean performance.
REPRESENTATIVE_SEEDS_MAP = {
    "WCE": 42,
    "LDAM": 88,
    "FocalLoss": 42,
    "FA_FL": 101
}


# ================================================================================
# SECTION 3: DATASET CLASS & DATA PREPARATION
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


def _get_class_counts(train_img_labels, class_names):
    class_counts = Counter(train_img_labels)
    return [class_counts[i] for i in range(len(class_names))]


def _calculate_class_weights(train_img_labels, class_names, device):
    class_counts = Counter(train_img_labels)
    sorted_counts = [class_counts[i] for i in range(len(class_names))]
    total_samples = sum(sorted_counts)
    weights = [total_samples / count if count > 0 else 0 for count in sorted_counts]
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]
    return torch.tensor(normalized_weights, dtype=torch.float32).to(device)


# --- MODIFIED DATA PREPARATION FUNCTION ---
def get_data_products(config, device, seed):
    """
    Orchestrates the data pipeline for a SINGLE SEED.
    """
    all_subjects, subjects_to_images, subject_to_original_label = _prepare_subject_data(config["DATA_DIR"],
                                                                                        config["ORIGINAL_CLASSES"])

    class_mapping = {
        config["ORIGINAL_CLASSES"].index("Non Demented"): config["NEW_CLASSES"].index("Non Demented"),
        config["ORIGINAL_CLASSES"].index("Very mild Dementia"): config["NEW_CLASSES"].index("Very Mild Demented"),
        config["ORIGINAL_CLASSES"].index("Mild Dementia"): config["NEW_CLASSES"].index("Demented"),
        config["ORIGINAL_CLASSES"].index("Moderate Dementia"): config["NEW_CLASSES"].index("Demented")
    }
    subject_to_new_label = {s: class_mapping[li] for s, li in subject_to_original_label.items()}
    all_new_labels = [subject_to_new_label[s] for s in all_subjects]

    dev_subjects, _, dev_labels, _ = train_test_split(all_subjects, all_new_labels, test_size=config["TEST_SET_RATIO"],
                                                      random_state=seed, stratify=all_new_labels)
    val_split_ratio = config["VAL_SET_RATIO"] / (1 - config["TEST_SET_RATIO"])
    _, _, _, _ = train_test_split(dev_subjects, dev_labels, test_size=val_split_ratio, random_state=seed,
                                  stratify=dev_labels)

    # We only need the test split for this evaluation
    _, test_subjects, _, _ = train_test_split(all_subjects, all_new_labels, test_size=config["TEST_SET_RATIO"],
                                              random_state=seed, stratify=all_new_labels)
    test_images, test_labels = _unpack_split_data(test_subjects, subjects_to_images, subject_to_new_label)

    val_test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_dataset = DementiaDataset(test_images, test_labels, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4,
                             pin_memory=True)

    return {"loaders": {"test": test_loader}, "class_names": config["NEW_CLASSES"]}


# ================================================================================
# SECTION 4: MODELS, LOSS, AND TRAINING (UNCHANGED)
# Contains all model and loss class definitions. Training function is kept for completeness.
# ================================================================================
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, s=30.0):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max(m_list) / min(m_list))
        m_list = torch.FloatTensor(m_list)
        self.register_buffer('m_list', m_list)
        self.s = s

    def forward(self, inputs, targets):
        index = torch.zeros_like(inputs, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        batch_m = torch.matmul(self.m_list[None, :].to(inputs.device), index.transpose(0, 1).float())
        batch_m = batch_m.view((-1, 1))
        x_m = inputs - batch_m
        output = torch.where(index, x_m, inputs)
        return F.cross_entropy(self.s * output, targets)


class StandardFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super(StandardFocalLoss, self).__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha.gather(0, targets)
        focal_loss = alpha_t * torch.pow(1 - pt, self.gamma) * ce_loss
        return focal_loss.mean()


class FrequencyAdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha, gamma_base=2.0, lambda_val=1.0):
        super(FrequencyAdaptiveFocalLoss, self).__init__()
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


class DementiaModel(nn.Module):
    def __init__(self, num_classes):
        super(DementiaModel, self).__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        self.model.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features)
        return output


# ... (train_model, EarlyStopping, etc. can remain here if needed for 'train' mode)


# ================================================================================
# SECTION 5: NEW EVALUATION AND PLOTTING FUNCTIONS
# ================================================================================

def get_predictions(model, data_loader, device):
    """Helper function to get true labels and model probabilities for a dataset."""
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


# Add this import at the top of your script with the other sklearn imports
from sklearn.preprocessing import label_binarize


def generate_combined_representative_curves(all_models, all_data, representative_seeds_map, class_names, device):
    """
    Generates and saves the final ROC and PR curve plots comparing all models.
    This function plots the curve of the single "representative run" for each model,
    but calculates and displays the mean +/- std. dev. AUC/AP across all 5 runs in the legend.
    """
    print("\n===== GENERATING COMBINED REPRESENTATIVE ROC & PR CURVES =====")
    n_classes = len(class_names)

    # Initialize plots
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))

    model_types = list(all_models.keys())
    colors = cycle(['cornflowerblue', 'darkorange', 'forestgreen', 'red', 'purple'])

    for model_type, color in zip(model_types, colors):
        print(f"--- Processing model: {model_type} ---")

        # --- 1. Get the curve for the single representative run ---
        rep_seed = representative_seeds_map[model_type]
        print(f"  Using representative seed: {rep_seed}")

        rep_model = all_models[model_type][rep_seed]
        rep_test_loader = all_data[rep_seed]['loaders']['test']

        y_true_rep, y_probas_rep = get_predictions(rep_model, rep_test_loader, device)

        # --- 2. Calculate Mean/Std of metrics across ALL 5 seeds ---
        roc_aucs_all_runs = defaultdict(list)
        avg_precisions_all_runs = defaultdict(list)

        for seed in CONFIG["SEEDS"]:
            model = all_models[model_type][seed]
            test_loader = all_data[seed]['loaders']['test']
            y_true, y_probas = get_predictions(model, test_loader, device)

            # --- FIX: Binarize y_true for multiclass AUC/AP calculation ---
            y_true_binarized = label_binarize(y_true, classes=range(n_classes))

            # This handles if a class is missing in a split, though less likely with stratified splits
            if y_true_binarized.shape[1] < n_classes:
                # Pad with zeros if a class is not present in this split's test set
                padded_binarized = np.zeros((y_true_binarized.shape[0], n_classes))
                padded_binarized[:, :y_true_binarized.shape[1]] = y_true_binarized
                y_true_binarized = padded_binarized

            roc_aucs_all_runs['micro'].append(roc_auc_score(y_true_binarized, y_probas, average='micro'))
            avg_precisions_all_runs['micro'].append(
                average_precision_score(y_true_binarized, y_probas, average='micro'))

        # --- 3. Plot ROC Curve for the representative run ---
        mean_roc_auc = np.mean(roc_aucs_all_runs['micro'])
        std_roc_auc = np.std(roc_aucs_all_runs['micro'])

        # --- FIX: Binarize y_true_rep before calling roc_curve ---
        y_true_rep_binarized = label_binarize(y_true_rep, classes=range(n_classes))

        fpr, tpr, _ = roc_curve(y_true_rep_binarized.ravel(), y_probas_rep.ravel())
        ax_roc.plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_type} (AUC = {mean_roc_auc:.2f} ± {std_roc_auc:.2f})')

        # --- 4. Plot PR Curve for the representative run ---
        mean_ap = np.mean(avg_precisions_all_runs['micro'])
        std_ap = np.std(avg_precisions_all_runs['micro'])

        precision, recall, _ = precision_recall_curve(y_true_rep_binarized.ravel(), y_probas_rep.ravel())
        ax_pr.plot(recall, precision, color=color, lw=2,
                   label=f'{model_type} (AP = {mean_ap:.2f} ± {std_ap:.2f})')

    # --- Finalize and save ROC plot ---
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    ax_roc.set_xlim([0.0, 1.0]);
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate');
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Representative ROC Curves (Micro-Average)')
    ax_roc.legend(loc="lower right");
    ax_roc.grid(True)
    fig_roc.savefig("combined_representative_roc_curves.png");
    plt.close(fig_roc)
    print("-> Saved 'combined_representative_roc_curves.png'")

    # --- Finalize and save PR plot ---
    ax_pr.set_xlim([0.0, 1.0]);
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.set_xlabel('Recall');
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Representative Precision-Recall Curves (Micro-Average)')
    ax_pr.legend(loc="best");
    ax_pr.grid(True)
    fig_pr.savefig("combined_representative_pr_curves.png");
    plt.close(fig_pr)
    print("-> Saved 'combined_representative_pr_curves.png'")

# ================================================================================
# SECTION 6: MAIN EXECUTION BLOCK
# ================================================================================
if __name__ == '__main__':

    if CONFIG["MODE"] == "evaluate":
        # --- EVALUATION MODE ---
        print("===== RUNNING IN EVALUATION MODE =====")

        # Step 1: Prepare data for all seeds and cache it
        print("\n--- Preparing datasets for all seeds... ---")
        all_data = {}
        for seed in CONFIG["SEEDS"]:
            print(f"  Generating data split for seed: {seed}")
            all_data[seed] = get_data_products(CONFIG, CONFIG["DEVICE"], seed=seed)
        print("--- All datasets prepared. ---\n")

        # Step 2: Load all pre-trained models into memory
        print("--- Loading all pre-trained models... ---")
        all_models = defaultdict(dict)
        num_classes = len(CONFIG["NEW_CLASSES"])

        for loss_type, seed_map in WEIGHT_PATHS.items():
            for seed, path in seed_map.items():
                if not os.path.exists(path):
                    print(f"  [WARNING] Weight file not found for {loss_type} (seed {seed}): {path}. Skipping.")
                    continue

                print(f"  Loading model for {loss_type} (seed {seed})")
                model = DementiaModel(num_classes=num_classes)
                model.load_state_dict(torch.load(path, map_location=CONFIG["DEVICE"]))
                model.to(CONFIG["DEVICE"])
                all_models[loss_type][seed] = model
        print("--- All models loaded. ---\n")

        # Step 3: Generate the final combined plots
        generate_combined_representative_curves(
            all_models=all_models,
            all_data=all_data,
            representative_seeds_map=REPRESENTATIVE_SEEDS_MAP,
            class_names=CONFIG["NEW_CLASSES"],
            device=CONFIG["DEVICE"]
        )

    elif CONFIG["MODE"] == "train":
        # --- TRAINING MODE (Original script logic) ---
        print("===== RUNNING IN TRAINING MODE =====")
        # The original script's main execution block would go here.
        # This part is omitted for brevity as the focus is on evaluation.
        print("Training mode logic should be placed here.")

    else:
        raise ValueError(f"Invalid MODE in CONFIG: {CONFIG['MODE']}. Must be 'train' or 'evaluate'.")

    print("\n--- All Processes Complete ---")