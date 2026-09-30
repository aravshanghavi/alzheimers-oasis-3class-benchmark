import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class AlzheimerImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # 0 is a placeholder label


def create_dataloaders(root_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=20),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1.0, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}

    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            try:
                dataset = AlzheimerImageDataset(class_dir, transform=transform)
                if len(dataset) == 0:
                    print(f"Warning: No image files found in {class_dir}")
                    continue

                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                dataloaders[class_name] = dataloader
            except Exception as e:
                print(f"Error processing {class_name}: {str(e)}")

    return dataloaders


# Usage
root_directory = r"C:\Users\aravs\Desktop\Arav\Research\Divya Ma'am\Data"
dataloaders = create_dataloaders(root_directory)

# Print information about each dataloader
for class_name, dataloader in dataloaders.items():
    print(f"Class: {class_name}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Total images: {len(dataloader.dataset)}")
    print("---")
