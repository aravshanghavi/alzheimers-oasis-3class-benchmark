import os
import random
import math
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import multiprocessing

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

def create_dataloaders(root_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
                    dataset = MemoryEfficientImageDataset(class_dir, transform=transform, class_label=class_idx, mode=mode)
                    datasets[mode].append(dataset)
                print(f"Successfully processed {class_name}")
            except Exception as e:
                print(f"Error processing {class_name}: {str(e)}")

    train_dataset = torch.utils.data.ChainDataset(datasets['train'])
    val_dataset = torch.utils.data.ChainDataset(datasets['val'])
    test_dataset = torch.utils.data.ChainDataset(datasets['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader, test_loader

def estimate_dataset_size(dataloader):
    total_samples = 0
    for _ in dataloader:
        total_samples += 1
    return total_samples * dataloader.batch_size

def print_class_distribution(loader):
    class_counts = {}
    for _, labels in loader:
        for label in labels:
            if label.item() not in class_counts:
                class_counts[label.item()] = 1
            else:
                class_counts[label.item()] += 1
    print("Class distribution:")
    for class_idx, count in class_counts.items():
        print(f"Class {class_idx}: {count}")

def main():
    root_directory = r"C:\Users\aravs\Desktop\Arav\Research\Divya Ma'am\Data"
    train_loader, val_loader, test_loader = create_dataloaders(root_directory)

    print(f"Estimated Train set size: {estimate_dataset_size(train_loader)}")
    print(f"Estimated Validation set size: {estimate_dataset_size(val_loader)}")
    print(f"Estimated Test set size: {estimate_dataset_size(test_loader)}")

    print("\nTrain set:")
    print_class_distribution(train_loader)
    print("\nValidation set:")
    print_class_distribution(val_loader)
    print("\nTest set:")
    print_class_distribution(test_loader)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()