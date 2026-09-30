import os
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report

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

    train_dataset = DementiaDataset(train_data, train_labels, transform=transform)
    val_dataset = DementiaDataset(val_data, val_labels, transform=transform)
    test_dataset = DementiaDataset(test_data, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def class_count(train_labels, classes):

    class_counts = [train_labels.count(idx) for idx in range(len(classes))]
    class_weights = [1.0 / count for count in class_counts]
    class_weights = torch.tensor(class_weights).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights}")

    return class_count, class_weights

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

def model(classes, class_weights):
    weights = ResNet50_Weights.IMAGENET1K_V1
    resnet_model = resnet50(weights=weights)
    num_classes = len(classes)
    in_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = resnet_model.to(device)


    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)

    return resnet_model, criterion, optimizer

def train(resnet_model, train_loader, optimizer, criterion, val_loader):
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using a local path instead of Kaggle path
    save_model_path = os.path.join(os.getcwd(), "resnet_alzheimer_final.pth")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        resnet_model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        resnet_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = resnet_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=classes))

        # Saving the model with error handling
        if epoch == num_epochs - 1:
            try:
                torch.save(resnet_model.state_dict(), save_model_path)
                print(f"Model saved at: {save_model_path}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                # Fallback to saving in the current working directory with a timestamp
                import time
                fallback_path = f"resnet_model_{int(time.time())}.pth"
                try:
                    torch.save(resnet_model.state_dict(), fallback_path)
                    print(f"Model saved at fallback location: {fallback_path}")
                    save_model_path = fallback_path
                except Exception as e:
                    print(f"Failed to save model even at fallback location: {str(e)}")

    print("Training completed!")
    return resnet_model, save_model_path
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

if __name__=='__main__':

    train_data, val_data, test_data, train_labels, val_labels, test_labels, classes = load_data()
    class_count, class_weights = class_count(train_labels, classes)
    train_loader, val_loader, test_loader = create_data_loaders(train_data, train_labels, val_data, val_labels, test_data, test_labels, batch_size=32)
    resnet_model, criterion, optimizer = model(classes, class_weights)
    resnet_model, model_path = train(resnet_model, train_loader, optimizer, criterion, val_loader)
    evaluate_model(resnet_model, test_loader, classes)