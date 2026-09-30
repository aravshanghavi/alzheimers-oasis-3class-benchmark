import os
import re
from collections import defaultdict, Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================================
# SECTION 1: CONFIGURATION
# Centralized parameters for the data preparation pipeline.
# ================================================================================
CONFIG = {
    "DATA_DIR": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data",
    "RANDOM_STATE": 42,
    "TEST_SET_RATIO": 0.20,  # 20% of subjects for the final hold-out test set
    "VAL_SET_RATIO": 0.15,  # 15% of the *remaining* subjects for the validation set
    "BATCH_SIZE": 32,
    # Original class names exactly as they appear in the folder structure
    "ORIGINAL_CLASSES": ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"],
    # Definitive 3-class structure for the manuscript
    "NEW_CLASSES": ["Non Demented", "Very Mild Demented", "Demented"]
}


# ================================================================================
# SECTION 2: DATASET CLASS
# ================================================================================
class DementiaDataset(Dataset):
    """Standard Dataset class. No changes needed."""

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


# ================================================================================
# SECTION 3: INTERNAL HELPER FUNCTIONS
# These functions contain the core logic and are called by the main public function.
# ================================================================================
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
    subject_list = list(subjects_to_images.keys())
    return subject_list, subjects_to_images, subject_to_label


def _unpack_split_data(subject_list, subjects_to_images, subject_to_label):
    """(Internal) Expands a list of subject IDs into image paths and labels."""
    images, labels = [], []
    for subj in subject_list:
        subj_images = subjects_to_images[subj]
        images.extend(subj_images)
        labels.extend([subject_to_label[subj]] * len(subj_images))
    return images, labels


def _calculate_class_weights(train_img_labels, class_names, device):
    """(Internal) Calculates inverse frequency class weights for the training set."""
    print("\n===== STEP 5: CALCULATING CLASS WEIGHTS FOR LOSS FUNCTION =====")
    class_counts = Counter(train_img_labels)

    # Sort counts by class index (0, 1, 2) to ensure order
    sorted_counts = [class_counts[i] for i in range(len(class_names))]
    total_samples = sum(sorted_counts)

    # Method: Inverse Frequency. weight = 1 / (count / total)
    weights = [total_samples / count if count > 0 else 0 for count in sorted_counts]

    # Normalize weights to prevent them from becoming excessively large
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]

    weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)

    print("Image counts per class in training set (for weighting):")
    for i, count in enumerate(sorted_counts):
        print(f"  - Class '{class_names[i]}': {count} images")

    print(f"\nFinal calculated class weights (normalized):\n{weights_tensor}")
    return weights_tensor.to(device)


def _verify_dataloader_classes(data_loader, set_name, class_names):
    """(Internal) Iterates through a DataLoader to confirm all target classes are present."""
    print(f"\n--- Verifying class representation in '{set_name}' DataLoader ---")
    present_labels = set()
    for _, labels in data_loader:
        present_labels.update(labels.numpy())
    if len(present_labels) == len(class_names):
        print(f"SUCCESS: Found all {len(class_names)} classes.")
    else:
        print(f"WARNING: Found only {len(present_labels)} of {len(class_names)} classes.")


# ================================================================================
# SECTION 4: PUBLIC API FUNCTION
# This is the main, high-level function to be called from outside the module.
# ================================================================================
def get_data_loaders_and_weights(config, device):
    """
    Orchestrates the entire data loading, splitting, and preparation pipeline.

    Args:
        config (dict): A dictionary containing all necessary configuration parameters.
        device (torch.device): The device to which the class weights tensor will be sent.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader)
            - val_loader (DataLoader)
            - test_loader (DataLoader)
            - class_weights (torch.Tensor)
    """
    # ====== STEP 1: LOAD RAW DATA AND GROUP BY SUBJECT ======
    print("===== STEP 1: SCANNING DIRECTORIES AND GROUPING BY SUBJECT =====")
    all_subjects, subjects_to_images, subject_to_original_label = _prepare_subject_data(
        config["DATA_DIR"], config["ORIGINAL_CLASSES"]
    )
    print(f"Discovered {len(all_subjects)} unique subjects.")

    # ====== STEP 2: REMAP TO THE NEW 3-CLASS PROBLEM ======
    print("\n===== STEP 2: REMAPPING SUBJECTS TO THE 3-CLASS BENCHMARK =====")
    class_mapping = {
        config["ORIGINAL_CLASSES"].index("Non Demented"): config["NEW_CLASSES"].index("Non Demented"),
        config["ORIGINAL_CLASSES"].index("Very mild Dementia"): config["NEW_CLASSES"].index("Very Mild Demented"),
        config["ORIGINAL_CLASSES"].index("Mild Dementia"): config["NEW_CLASSES"].index("Demented"),
        config["ORIGINAL_CLASSES"].index("Moderate Dementia"): config["NEW_CLASSES"].index("Demented")
    }
    subject_to_new_label = {s: class_mapping[li] for s, li in subject_to_original_label.items()}
    all_new_labels = [subject_to_new_label[s] for s in all_subjects]
    print("Subject distribution for new 3-Class problem:")
    print(Counter([config["NEW_CLASSES"][l] for l in all_new_labels]))

    # ====== STEP 3: PERFORM STRICT, SUBJECT-LEVEL SPLIT ======
    print("\n===== STEP 3: PERFORMING STRATIFIED SUBJECT-LEVEL SPLIT =====")
    dev_subjects, test_subjects, dev_labels, _ = train_test_split(
        all_subjects, all_new_labels, test_size=config["TEST_SET_RATIO"],
        random_state=config["RANDOM_STATE"], stratify=all_new_labels
    )
    val_split_ratio = config["VAL_SET_RATIO"] / (1 - config["TEST_SET_RATIO"])
    train_subjects, val_subjects, _, _ = train_test_split(
        dev_subjects, dev_labels, test_size=val_split_ratio,
        random_state=config["RANDOM_STATE"], stratify=dev_labels
    )
    print(f"Split complete: {len(train_subjects)} Train, {len(val_subjects)} Val, {len(test_subjects)} Test subjects.")

    # ====== STEP 4: CREATE DATASETS AND DATALOADERS ======
    print("\n===== STEP 4: CREATING DATASETS AND DATALOADERS =====")
    train_images, train_labels = _unpack_split_data(train_subjects, subjects_to_images, subject_to_new_label)
    val_images, val_labels = _unpack_split_data(val_subjects, subjects_to_images, subject_to_new_label)
    test_images, test_labels = _unpack_split_data(test_subjects, subjects_to_images, subject_to_new_label)
    print(f"Image counts: {len(train_images)} Train, {len(val_images)} Val, {len(test_images)} Test.")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = DementiaDataset(train_images, train_labels, transform=train_transform)
    val_dataset = DementiaDataset(val_images, val_labels, transform=val_test_transform)
    test_dataset = DementiaDataset(test_images, test_labels, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4,
                             pin_memory=True)
    print("DataLoaders created successfully.")

    # ====== STEP 5: CALCULATE CLASS WEIGHTS ======
    class_weights = _calculate_class_weights(train_labels, config["NEW_CLASSES"], device)

    # ====== FINAL VERIFICATION ======
    print("\n===== FINAL VERIFICATION =====")
    _verify_dataloader_classes(train_loader, "Training", config["NEW_CLASSES"])
    _verify_dataloader_classes(val_loader, "Validation", config["NEW_CLASSES"])
    _verify_dataloader_classes(test_loader, "Test", config["NEW_CLASSES"])

    return train_loader, val_loader, test_loader, class_weights

class FrequencyAdaptiveFocalLoss(nn.Module):
    """
    Frequency-Adaptive Focal Loss (FA-FL).

    This loss function adapts the focusing parameter `gamma` for each class based on its
    pre-calculated balancing weight `alpha`. The hypothesis is that minority classes
    are inherently "harder" and require a more aggressive focus (a higher gamma).

    Args:
        alpha (torch.Tensor): A tensor of shape (num_classes,) containing the
                              normalized class-balancing weights (alphas).
        gamma_base (float): The baseline focusing parameter. Default: 2.0.
        lambda_val (float): A hyperparameter that controls how strongly the class
                            rarity (via alpha) influences the focus. Default: 1.0.
    """

    def __init__(self, alpha, gamma_base=2.0, lambda_val=1.0):
        super(FrequencyAdaptiveFocalLoss, self).__init__()

        if not isinstance(alpha, torch.Tensor):
            raise TypeError("alpha must be a torch.Tensor")

        self.register_buffer('alpha', alpha)

        # Calculate class-dependent gamma_t = gamma_base + lambda * alpha
        gammas = gamma_base + lambda_val * self.alpha
        self.register_buffer('gammas', gammas)

        print("--- Initializing Frequency-Adaptive Focal Loss ---")
        print(f"Base Gamma: {gamma_base}, Lambda: {lambda_val}")
        print(f"Class Alphas: {self.alpha.cpu().numpy()}")
        print(f"Resulting Gammas per class: {self.gammas.cpu().numpy()}")
        print("-------------------------------------------------")

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Raw, un-normalized scores from the model of
                                   shape (N, C), where N is batch size and C is num_classes.
            targets (torch.Tensor): Ground truth labels of shape (N,).

        Returns:
            torch.Tensor: The final computed loss.
        """
        # Calculate the per-element cross-entropy loss, which is -log(pt)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get the probabilities of the correct class, pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Gather the alpha and gamma values for each sample in the batch
        alpha_t = self.alpha.gather(0, targets)
        gamma_t = self.gammas.gather(0, targets)

        # This is the core of the FA-FL calculation
        # The modulating factor is alpha_t * (1 - pt)^gamma_t
        modulating_factor = alpha_t * torch.pow(1 - pt, gamma_t)

        # The final loss is the modulating_factor times the cross-entropy loss
        focal_loss = modulating_factor * ce_loss

        return focal_loss.mean()


# ================================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# Clean, high-level orchestration of the data pipeline.
# ================================================================================
if __name__ == '__main__':
    # Set the device for computation (and for the weights tensor)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {DEVICE}")
    print("-" * 60)

    # Call the main API function to get DataLoaders and class weights
    train_loader, val_loader, test_loader, class_weights = get_data_loaders_and_weights(CONFIG, DEVICE)

    print("-" * 60)
    print("PIPELINE COMPLETE.")
    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Val Loader:   {len(val_loader)} batches")
    print(f"Test Loader:  {len(test_loader)} batches")
    print(f"Class Weights Tensor on {class_weights.device}:\n{class_weights}")

    fa_loss_fn = FrequencyAdaptiveFocalLoss(alpha=class_weights, gamma_base=2.0, lambda_val=2.0)