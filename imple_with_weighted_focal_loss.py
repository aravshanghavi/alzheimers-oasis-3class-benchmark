# Standard library imports
import os
import random
import math
import warnings
import multiprocessing
from scipy.ndimage import gaussian_filter, map_coordinates

# Third-party imports
import numpy as np
from PIL import Image

# PyTorch and related imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.ndimage import map_coordinates
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms, models

# Machine learning imports
import timm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

class MemoryEfficientImageDataset(IterableDataset):
    def __init__(self, img_dir, transform=None, class_label=0, mode='train', split_ratios=(0.9, 0.02, 0.08)):
        self.img_dir = img_dir
        self.transform = transform
        self.class_label = class_label
        self.mode = mode
        self.split_ratios = split_ratios
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(self.images)
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        dataset_size = len(self.images)
        train_split = int(self.split_ratios[0] * dataset_size)
        val_split = int(self.split_ratios[1] * dataset_size)

        if mode == 'train':
            self.start_idx, self.end_idx = 0, train_split
        elif mode == 'val':
            self.start_idx, self.end_idx = train_split, train_split + val_split
        else:  # test
            self.start_idx, self.end_idx = train_split + val_split, dataset_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start_idx
            iter_end = self.end_idx
        else:
            per_worker = int(math.ceil((self.end_idx - self.start_idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start_idx + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end_idx)

        for idx in range(iter_start, iter_end):
            img_path = os.path.join(self.img_dir, self.images[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            yield image, self.class_label

    def __len__(self):
        return self.end_idx - self.start_idx


def collect_all_labels(root_dir):
    """Collect all labels from the dataset for computing class weights"""
    labels = []
    for class_idx, class_name in enumerate(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            # Count number of images in this class directory
            num_images = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            labels.extend([class_idx] * num_images)
    return np.array(labels)


def compute_class_weights(root_dir, device='cuda'):
    """Compute balanced class weights for the entire dataset"""
    labels = collect_all_labels(root_dir)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights).to(device)

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

class TransFuseModel(nn.Module):
    def __init__(self, num_classes):
        super(TransFuseModel, self).__init__()
        self.feature_extractor = models.resnet34(weights='IMAGENET1K_V1')
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor.fc = nn.Identity()

        self.transformer = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=1)
        self.transformer.head = nn.Identity()

        self.fusion = nn.Sequential(
            nn.Conv2d(768 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        self.f2_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.t2_head = nn.Linear(768, num_classes)
        self.f0_head = nn.Linear(256 * 7 * 7, num_classes)

    def forward(self, x):
        # CNN branch

        cnn_features = self.feature_extractor.conv1(x)
        cnn_features = self.feature_extractor.bn1(cnn_features)
        cnn_features = self.feature_extractor.relu(cnn_features)
        cnn_features = self.feature_extractor.maxpool(cnn_features)

        cnn_features = self.feature_extractor.layer1(cnn_features)
        cnn_features = self.feature_extractor.layer2(cnn_features)
        cnn_features = self.feature_extractor.layer3(cnn_features)
        cnn_features = self.feature_extractor.layer4(cnn_features)
        #print(cnn_features.shape)
        cnn_features = cnn_features.view(cnn_features.size(0), 512, 7, 7)
        #print(cnn_features.shape)
        output = self.classifier(cnn_features.view(cnn_features.size(0), -1))
        return output
        # Transformer branch
        #transformer_features = self.transformer(x)
        #transformer_features = transformer_features.view(transformer_features.size(0), 768, 1, 1).expand(-1, -1, 7, 7)

        # Fusion
        #fused_features = torch.cat([cnn_features, transformer_features], dim=1)
        #fused_features = self.fusion(fused_features)

        # Main output
        #print(fused_features.shape)
        #output = self.classifier(fused_features.view(fused_features.size(0), -1))
        #print(output.shape)
        # Auxiliary outputs
        #f2_out = self.f2_head(fused_features)
        #t2_out = self.t2_head(transformer_features.mean(dim=[2, 3]))
        #f0_out = self.f0_head(fused_features.view(fused_features.size(0), -1))

        #return output, f2_out, t2_out, f0_out


class CombinedWeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, device, alpha=0.5, gamma=2.0):
        super(CombinedWeightedFocalLoss, self).__init__()
        self.class_weights = class_weights.to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.alpha = alpha  # weight between focal loss and cross entropy
        self.gamma = gamma  # focusing parameter for focal loss

    def forward(self, inputs, targets):
        # Calculate the combined loss (weighted focal loss + weighted cross-entropy)
        combined_loss = self.compute_combined_loss(inputs, targets)
        return combined_loss

    def compute_combined_loss(self, output, targets):
        # Compute Weighted Cross Entropy Loss
        ce_loss = self.ce_loss(output, targets)

        # Compute Weighted Focal Loss
        log_probs = F.log_softmax(output, dim=1)
        targets_prob = F.one_hot(targets, num_classes=output.size(1)).float().to(output.device)

        # Apply class weights to the one-hot encoded targets
        weighted_targets = targets_prob * self.class_weights.unsqueeze(0)

        # Compute the focal term (1 - p_t)^gamma
        probs = log_probs.exp()
        focal_weight = (1 - probs) ** self.gamma

        # Calculate weighted focal loss
        focal_loss = -(focal_weight * log_probs * weighted_targets).sum(dim=1).mean()

        # Combine weighted cross-entropy and focal loss
        combined_loss = (1 - self.alpha) * ce_loss + self.alpha * focal_loss

        return combined_loss


# Function to create the combined weighted focal loss
def create_weighted_focal_loss(class_weights, device, alpha=0.45, gamma=3.0):
    return CombinedWeightedFocalLoss(class_weights, device, alpha=alpha, gamma=gamma)


def train_model(model, train_loader, val_loader, custom_criterion, num_epochs=20, learning_rate=1e-4,
                accumulation_steps=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = custom_criterion
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3, amsgrad=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.7)
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(patience=7)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        samples_processed = 0
        batch_count = 0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            current_batch_size = inputs.size(0)
            samples_processed += current_batch_size
            batch_count += 1

            with torch.amp.autocast('cuda'):  # Updated from torch.cuda.amp.autocast()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (i+1)%1000==0:  # Print every 100 batches
                print(f"Processed {i+1} / ({len(train_loader)} batches")

            running_loss += loss.item() * current_batch_size
            running_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        epoch_loss = running_loss / samples_processed
        epoch_acc = running_corrects.double() / samples_processed

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                current_batch_size = inputs.size(0)
                val_samples += current_batch_size

                outputs = model(inputs)
                #output, f2_out, t2_out, f0_out = outputs
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_running_loss += loss.item() * current_batch_size
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / val_samples
        val_acc = float(val_running_corrects) / val_samples

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print()

    print(f'Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def calculate_top3_error(model, dataloader, device):
    model.eval()
    all_labels = []
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(inputs)
            _, preds = torch.topk(output, 3, dim=1)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            correct += np.sum(np.any(preds == labels[:, None], axis=1))
            all_labels.extend(labels)
    top3_error = 1 - correct / len(all_labels)
    print(f'Top-3 Error: {top3_error:.4f}')


def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    test_running_corrects = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(inputs)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_running_corrects += torch.sum(preds == labels.data)
    accuracy = (test_running_corrects / len(dataloader.dataset))*100
    print(f"Overall Accuracy on Test Set: {accuracy:.4f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names))

class ElasticTransform:
    """Elastic deformation of images as described in [Simard2003]."""

    def __init__(self, alpha=1000, sigma=30):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        shape = img.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distorted_image = map_coordinates(img, indices, order=1, mode='reflect')
        return Image.fromarray(distorted_image.reshape(shape).astype(np.uint8))


class RandomMRIIntensityShift:
    def __init__(self, intensity_range=(-0.1, 0.1)):
        self.intensity_range = intensity_range

    def __call__(self, img):
        shift = random.uniform(*self.intensity_range)
        img_array = np.array(img).astype(np.float32)
        img_array = img_array + (img_array * shift)
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))


class RandomMRIIntensityScale:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = random.uniform(*self.scale_range)
        img_array = np.array(img).astype(np.float32)
        img_array = img_array * scale
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))


class RandomIntensityNoise:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            noise = torch.randn_like(img) * self.noise_std
            return torch.clamp(img + noise, 0, 1)
        else:
            img_array = np.array(img).astype(np.float32)
            noise = np.random.normal(0, self.noise_std, img_array.shape)
            img_array = img_array + (img_array * noise)
            img_array = np.clip(img_array, 0, 255)
            return Image.fromarray(img_array.astype(np.uint8))


def create_dataloaders(root_dir, batch_size=8, img_size=224):
    """Create train, validation, and test dataloaders."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),

        # Spatial transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),

        # MRI-specific augmentations
        RandomMRIIntensityShift(intensity_range=(-0.2, 0.2)),
        RandomMRIIntensityScale(scale_range=(0.85, 1.15)),
        ElasticTransform(alpha=1000, sigma=30),
        RandomIntensityNoise(noise_std=0.1),

        transforms.RandomCrop(224, padding=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.165], std=[0.1785])
    ])

    # Validation/Test transforms without augmentations
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.165], std=[0.1785])
    ])

    datasets = {
        'train': [],
        'val': [],
        'test': []
    }

    for class_idx, class_name in enumerate(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            try:
                for mode in ['train', 'val', 'test']:
                    current_transform = transform if mode == 'train' else val_transform
                    dataset = MemoryEfficientImageDataset(
                        class_dir,
                        transform=current_transform,
                        class_label=class_idx,
                        mode=mode
                    )
                    datasets[mode].append(dataset)
                print(f"Successfully processed {class_name}")
            except Exception as e:
                print(f"Error processing {class_name}: {str(e)}")

    train_dataset = torch.utils.data.ChainDataset(datasets['train'])
    val_dataset = torch.utils.data.ChainDataset(datasets['val'])
    test_dataset = torch.utils.data.ChainDataset(datasets['test'])

    # Create dataloaders with reduced number of workers for better stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def estimate_dataset_size(dataloader):
    total_samples = 0
    #for _ in dataloader:
    #    total_samples += 1
    print(len(dataloader))
    return len(dataloader) * dataloader.batch_size

def compute_alpha_from_class_weights(class_weights):
    """
    Convert class weights to alpha values for focal loss.
    Higher class weights should correspond to higher alpha values.
    """
    # Normalize class weights to sum to 1
    normalized_weights = class_weights / class_weights.sum()

    # Scale to a reasonable range for alpha (e.g., 0.25 to 0.75)
    alpha_min, alpha_max = 0.25, 0.75
    alpha = alpha_min + (alpha_max - alpha_min) * normalized_weights

    return alpha


def main():
    warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")
    root_directory = r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data"

    class_names = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]

    # Compute class weights
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = compute_class_weights(root_directory, device)
    print("\nClass weights:")
    for idx, weight in enumerate(class_weights):
        print(f"Class {idx}: {weight.item():.4f}")

    print("Making dataloaders")
    train_loader, val_loader, test_loader = create_dataloaders(root_directory)
    print("Made dataloaders")

    print(f"\nEstimated Train set size: {estimate_dataset_size(train_loader)}")
    print(f"Estimated Validation set size: {estimate_dataset_size(val_loader)}")
    print(f"Estimated Test set size: {estimate_dataset_size(test_loader)}")

    alpha_weights = compute_alpha_from_class_weights(class_weights)
    aux_weight = 0.3
    custom_criterion = create_weighted_focal_loss(
        class_weights=class_weights,
        device=device,
        alpha=0.3)

    print("\nClass weights and corresponding alpha values:")
    for idx, (weight, alpha) in enumerate(zip(class_weights, alpha_weights)):
        print(f"Class {idx}: Weight = {weight.item():.4f}, Alpha = {alpha.item():.4f}")

    model = TransFuseModel(num_classes=len(class_weights)).to(device)
    print(f" Model parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / (10 ** 6)}")
    trained_model = train_model(model, train_loader, val_loader, custom_criterion=custom_criterion, num_epochs=20,
                                learning_rate=8e-5, accumulation_steps=2)

    print("Evaluating TransFuse model on the test set...")
    evaluate_model(trained_model, test_loader, class_names)
    calculate_top3_error(trained_model, test_loader, device)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()