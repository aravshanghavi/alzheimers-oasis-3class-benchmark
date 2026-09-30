import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim as optim
from torch.optim import lr_scheduler
import seaborn as sns
import timm
import numpy as np

# XAI Imports
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz


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


# 1. Enhanced Feature Extraction

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        """
        Residual block that helps maintain gradient flow and feature preservation
        Args:
            dim (int): Input dimension of the features
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),  # First linear transformation
            nn.LayerNorm(dim),  # Normalize the features
            nn.ReLU(),  # Non-linear activation
            nn.Dropout(0.2),  # Regular dropout for regularization
            nn.Linear(dim, dim),  # Second linear transformation
            nn.LayerNorm(dim)  # Final normalization
        )

    def forward(self, x):
        # Add the input to the transformed features (skip connection)
        return x + self.block(x)


class EnhancedDementiaModel(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedDementiaModel, self).__init__()
        # Load pretrained ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model_resnet = resnet50(weights=weights)  # Renamed to avoid conflict
        self.model_resnet = nn.Sequential(*list(self.model_resnet.children())[:-1])

        # Load pretrained ViT
        self.transformer = timm.create_model('vit_base_patch16_224',
                                             pretrained=True,
                                             in_chans=3)
        self.transformer.head = nn.Identity()

        # Attention module for enhanced feature interaction
        self.attention = nn.Sequential(
            nn.Linear(2816, 2816),
            nn.LayerNorm(2816),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(1024),
            ResidualBlock(1024),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet feature extraction
        x_resnet = self.model_resnet(x)
        x_resnet = torch.flatten(x_resnet, 1)  # [batch_size, 2048]

        # ViT feature extraction
        x_vit = self.transformer(x)  # [batch_size, 768]

        # Concatenate features
        features = torch.cat([x_resnet, x_vit], dim=1)  # [batch_size, 2816]

        # Apply attention
        attended = self.attention(features)
        features = features + attended  # Residual connection for attention

        # Classification with residual blocks
        output = self.classifier(features)

        return output


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


def train_model(model, train_loader, val_loader, criterion, optimizer, classes, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    accumulation_steps = 2
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    save_model_path = "resnet_alzheimer_final.pth"
    early_stopping = EarlyStopping(patience=5, verbose=True)  # Added verbose
    best_model_wts = model.state_dict().copy()  # Use .copy()
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    # Removed redundant running_loss and running_corrects initialization here

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0  # Initialized here
        running_corrects = 0  # Initialized here
        optimizer.zero_grad()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1} Training")
        for i, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # Normalize loss to account for accumulation
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * images.size(0) * accumulation_steps  # Adjust for normalized loss
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix({'loss': running_loss / ((i + 1) * train_loader.batch_size),
                                      'acc': running_corrects.double() / ((i + 1) * train_loader.batch_size)})

        # scheduler.step() # Step per epoch
        avg_train_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        val_progress_bar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch + 1} Validation")
        with torch.no_grad():
            for inputs, labels in val_progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_progress_bar.set_postfix({'val_loss': val_running_loss / len(val_loader.dataset),
                                              'val_acc': float(val_running_corrects) / len(val_loader.dataset)})

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = float(val_running_corrects) / len(val_loader.dataset)  # Removed *100

        print(
            f"\nTrain Loss: {avg_train_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict().copy()  # Use .copy()
            print(f"New best validation accuracy: {best_acc:.4f}")
            torch.save(model.state_dict(), save_model_path)  # Save best model

        # evaluate_model(model, val_loader, classes, set_name="val_epoch_" + str(epoch+1)) # Changed set to set_name
        scheduler.step()  # Step per epoch after validation

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print(f"Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), save_model_path) # Already saved when best_acc improves
    return model


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

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = DementiaDataset(train_data, train_labels, transform=train_transform)
    val_dataset = DementiaDataset(val_data, val_labels)
    test_dataset = DementiaDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


class BalancedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2.0, class_thresholds=None):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha  # Per-class alpha can also be used if needed
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)

        # Additional weighting for the smallest class
        self.class_thresholds = class_thresholds if class_thresholds is not None else {}

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Basic focal loss calculation
        # Alpha can be a scalar or a tensor for class-specific alpha
        if isinstance(self.alpha, float):
            alpha_t = self.alpha * torch.ones_like(targets, dtype=torch.float32).to(inputs.device)
            alpha_t[targets == 0] = 1 - self.alpha  # Example if alpha is for positive class
        elif isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha[targets]
        else:  # list or array
            alpha_t = torch.tensor(self.alpha, dtype=torch.float32).to(inputs.device)[targets]

        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # Add class-specific weights
        applied_class_weights = self.class_weights[targets]

        # Add extra weighting for specific classes based on thresholds
        extra_weights = torch.ones_like(targets, dtype=torch.float32).to(inputs.device)
        for class_idx, multiplier in self.class_thresholds.items():
            extra_weights[targets == class_idx] *= multiplier

        # Combine all weighting factors
        final_weights = focal_weight * applied_class_weights * extra_weights
        loss = final_weights * ce_loss
        return loss.mean()


class EnhancedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha=0.25, gamma=2.0, class_thresholds=None, boundary_classes=None,
                 boundary_weight=0.5):
        super(EnhancedFocalLoss, self).__init__()
        self.base_focal = BalancedFocalLoss(class_weights, alpha, gamma, class_thresholds)
        self.boundary_classes = boundary_classes  # e.g., {3: 2} for Very Mild (idx 3) and Non-Demented (idx 2)
        self.boundary_weight = boundary_weight
        self.boundary_margin = 1.0  # Margin for boundary loss

    def forward(self, inputs, targets):
        focal_loss = self.base_focal(inputs, targets)
        boundary_loss_val = 0.0

        if self.boundary_classes:
            softmax_probs = F.softmax(inputs, dim=1)
            for class1_idx, class2_idx in self.boundary_classes.items():
                # Consider samples belonging to either class for this boundary loss
                mask_class1 = (targets == class1_idx)
                mask_class2 = (targets == class2_idx)

                # Probabilities for class1 and class2
                probs_c1 = softmax_probs[:, class1_idx]
                probs_c2 = softmax_probs[:, class2_idx]

                # Difference in probabilities for samples belonging to these classes
                # We want to push their probabilities further apart for their respective true classes
                # For samples of class1, we want (prob_c1 - prob_c2) to be large
                # For samples of class2, we want (prob_c2 - prob_c1) to be large
                # This can be simplified to minimizing the probability of the "other" class in the pair

                # Let's try a margin-based loss: encourage separation
                # For class1 true samples, penalize if prob_c1 < prob_c2 + margin
                loss_c1_boundary = torch.relu(self.boundary_margin - (
                            probs_c1[mask_class1] - probs_c2[mask_class1])).mean() if mask_class1.any() else 0.0
                # For class2 true samples, penalize if prob_c2 < prob_c1 + margin
                loss_c2_boundary = torch.relu(self.boundary_margin - (
                            probs_c2[mask_class2] - probs_c1[mask_class2])).mean() if mask_class2.any() else 0.0

                current_boundary_loss = (loss_c1_boundary + loss_c2_boundary) / 2.0
                boundary_loss_val += current_boundary_loss

        total_loss = focal_loss + self.boundary_weight * boundary_loss_val
        return total_loss


def create_weighted_focal_loss(class_weights_tensor, classes, alpha_config=0.25,
                               gamma=2.0):  # Renamed class_weights to class_weights_tensor
    # Increase gamma for more focus on hard examples
    gamma_val = 3.0  # Renamed

    # Calculate inverse class frequency for thresholding
    # This part seems to duplicate the class_weights calculation logic.
    # Assuming class_weights_tensor are already inverse frequency or balanced.

    # Example: "Non Demented" is index 2, "Very mild Dementia" is index 3
    # Based on your classes: ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
    # Indices: Mild=0, Moderate=1, Non=2, VeryMild=3

    # Set higher threshold for classes with very low frequency (e.g., Moderate Dementia)
    class_thresholds = {}
    # Assuming class_weights_tensor reflects counts or inverse frequencies.
    # Let's find the index of "Moderate Dementia"
    try:
        moderate_idx = classes.index("Moderate Dementia")
        class_thresholds[moderate_idx] = 5.0  # Extra weight for Moderate Dementia
    except ValueError:
        print("Warning: 'Moderate Dementia' not in classes. Threshold not applied.")

    # Alpha can be a list/tensor matching number of classes
    # For example, give more weight to under-represented positive classes if alpha < 0.5
    # Or use a scalar alpha as before.
    # Example: alpha_values = [0.25, 0.35, 0.25, 0.30] where 0.35 for Moderate, 0.30 for Very Mild
    alpha_values = alpha_config  # Using the passed alpha_config

    # Define which classes are hard to distinguish
    boundary_pairs = {}
    try:
        non_demented_idx = classes.index("Non Demented")
        very_mild_idx = classes.index("Very mild Dementia")
        boundary_pairs[very_mild_idx] = non_demented_idx  # Penalize confusion between Very Mild and Non-Demented
    except ValueError:
        print("Warning: 'Non Demented' or 'Very mild Dementia' not in classes. Boundary loss for them not applied.")

    return EnhancedFocalLoss(
        class_weights=class_weights_tensor,
        alpha=alpha_values,
        gamma=gamma_val,
        class_thresholds=class_thresholds,
        boundary_classes=boundary_pairs,
        boundary_weight=0.3  # Adjusted weight for boundary loss
    )


def evaluate_model(model, test_loader, classes, set_name=""):  # Changed set to set_name
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    total = 0

    eval_progress_bar = tqdm(test_loader, total=len(test_loader), desc=f"Evaluating on {set_name} set")
    with torch.no_grad():
        for images, labels in eval_progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            eval_progress_bar.set_postfix({'acc': (test_correct / total) * 100})

    accuracy = (test_correct / total) * 100
    print(f"\nOverall {set_name.capitalize()} Accuracy: {accuracy:.2f}%")

    # Ensure target_names has the correct number of elements
    if len(classes) == np.max(all_labels) + 1 and len(classes) == np.max(all_preds) + 1:
        report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
        print(report)
    else:
        print("Warning: Mismatch in number of classes for classification report.")
        print(f"Num classes: {len(classes)}, Max label: {np.max(all_labels)}, Max pred: {np.max(all_preds)}")
        report = classification_report(all_labels, all_preds, zero_division=0)
        print(report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))  # Ensure all classes are represented

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title(f'Confusion Matrix ({set_name} set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha="right")  # ha for alignment
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the confusion matrix plot
    plt.savefig(f'confusion_matrix_{set_name}.png')
    print(f"Confusion matrix saved to confusion_matrix_{set_name}.png")
    plt.close()

    return accuracy, cm


def generate_saliency_maps(model, data_loader, classes, device, num_images=5):
    """
    Generates and displays saliency maps for a few images using Integrated Gradients.
    """
    model.eval()
    ig = IntegratedGradients(model)

    images_so_far = 0

    # Inverse normalization for visualization
    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean_nums, std_nums)],
        std=[1 / s for s in std_nums]
    )

    print(f"\nGenerating Saliency Maps for {num_images} images...")

    data_iter = iter(data_loader)
    for _ in range(num_images):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            print("Reached end of data loader while generating saliency maps.")
            break

        img_tensor = images[0:1].to(device)  # Take the first image of the batch
        true_label_idx = labels[0].item()

        img_tensor.requires_grad = True  # Important for Captum

        # Get model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_idx = predicted_idx.item()

        # Generate attributions using Integrated Gradients
        # Baseline can be a black image (tensor of zeros)
        baselines = torch.zeros_like(img_tensor)
        attributions_ig, delta = ig.attribute(img_tensor, baselines=baselines, target=predicted_idx,
                                              return_convergence_delta=True)

        # Process for visualization
        img_to_show = inv_normalize(img_tensor.squeeze().cpu().detach()).permute(1, 2, 0).numpy()
        attr_to_show = attributions_ig.squeeze().cpu().detach().permute(1, 2, 0).numpy()

        print(f"\nImage {images_so_far + 1}:")
        print(f"  True Label: {classes[true_label_idx]}, Predicted Label: {classes[predicted_idx]}")
        print(f"  Integrated Gradients Convergence Delta: {delta.item():.4f}")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_to_show)
        axs[0].set_title(f"Original Image\nTrue: {classes[true_label_idx]}")
        axs[0].axis('off')

        # Visualize attributions
        vis_obj = viz.visualize_image_attr(attr_to_show,
                                           img_to_show,
                                           method="blended_heat_map",
                                           sign="all",  # positive, negative, absolute_value, all
                                           show_colorbar=True,
                                           title=f"Saliency Map (IG)\nPredicted: {classes[predicted_idx]}",
                                           use_pyplot=False)  # Important: use_pyplot=False for subplots

        axs[1].imshow(vis_obj.fig.axes[0].images[0].get_data())  # Get data from viz object
        # If using blended_heat_map, it might draw its own colorbar.
        # axs[1].images[0].set_cmap('hot') # Example: setting a colormap if not blended
        axs[1].set_title(f"Saliency Map (IG)\nPredicted: {classes[predicted_idx]}")
        axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"saliency_map_image_{images_so_far + 1}.png")
        print(f"Saliency map saved to saliency_map_image_{images_so_far + 1}.png")
        plt.show()  # Display the plot
        plt.close(fig)

        images_so_far += 1
        if images_so_far >= num_images:
            break
    if images_so_far == 0:
        print("Could not generate any saliency maps. Check data_loader and num_images.")


if __name__ == '__main__':
    # Load and prepare data
    try:
        train_data, val_data, test_data, train_labels, val_labels, test_labels, classes = load_data()
    except ValueError as e:
        print(f"Error during data loading: {e}")
        exit()

    if not train_data:  # Check if data loading was successful
        print("No training data loaded. Exiting.")
        exit()

    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size=16
        # Reduced batch size for potential memory constraints
    )

    _, class_weights = class_count(train_labels, classes)  # Use the renamed function and get the tensor

    # Initialize model and training components
    learning_rate = 1e-4  # Adjusted learning rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EnhancedDementiaModel(num_classes=len(classes))
    criterion = create_weighted_focal_loss(class_weights_tensor=class_weights, classes=classes,
                                           alpha_config=0.25)  # Pass the tensor
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Adjusted weight decay

    # Train model
    print("Starting model training...")
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, classes,
                                num_epochs=20)  # num_epochs can be adjusted

    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_accuracy, confusion_mat = evaluate_model(trained_model, test_loader, classes, set_name="test")
    print("\nTest Set Confusion Matrix:")
    print(confusion_mat)

    # Generate Saliency Maps for XAI
    print("\nGenerating XAI Saliency Maps for test images...")
    # We need a data loader that doesn't shuffle for consistent XAI examples if run multiple times, test_loader is fine.
    generate_saliency_maps(trained_model, test_loader, classes, device, num_images=5)

    print("\n--- Script Finished ---")