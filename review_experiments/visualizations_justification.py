import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os


def generate_sequential_augmentation_panel(class_name, img_path):
    """
    Loads a single image and creates a panel showing the step-by-step
    application of the full training augmentation pipeline.
    """
    print(f"Generating augmentation panel for class: {class_name}...")

    # 1. Define the individual transformation steps
    resize_transform = transforms.Resize((224, 224))
    affine_transform = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
    jitter_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2)
    hflip_transform = transforms.RandomHorizontalFlip(p=1.0)  # p=1.0 to ensure it happens for viz
    vflip_transform = transforms.RandomVerticalFlip(p=1.0)  # p=1.0 to ensure it happens for viz

    to_tensor_transform = transforms.ToTensor()
    normalize_transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # Un-normalize for visualization purposes
    un_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )

    # 2. Load the original image
    original_img = Image.open(img_path).convert("RGB")

    # 3. Apply transforms sequentially
    img_resized = resize_transform(original_img)
    img_affine = affine_transform(img_resized)
    img_jitter = jitter_transform(img_affine)
    img_hflip = hflip_transform(img_jitter)
    img_vflip = vflip_transform(img_hflip)

    # For the final step, apply the full pipeline to show a random final result
    final_tensor = normalize_transform(to_tensor_transform(img_vflip))
    final_img_for_viz = transforms.ToPILImage()(un_normalize(final_tensor))

    # 4. Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Augmentation Pipeline for Class: '{class_name}'", fontsize=16)

    images_to_plot = [
        (original_img, "1. Original"),
        (img_resized, "2. Resized (224x224)"),
        (img_affine, "3. After Random Affine\n(Rotate, Scale, Shear)"),
        (img_jitter, "4. After Color Jitter\n(Brightness, Contrast)"),
        (img_vflip, "5. After Random Flips\n(Horizontal & Vertical)"),
        (final_img_for_viz, "6. Final Normalized Result\n(Shown Un-normalized)")
    ]

    for i, (img, title) in enumerate(images_to_plot):
        ax = axes.flat[i]
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Sanitize class name for use in filename
    safe_class_name = class_name.replace(" ", "_")
    save_path = f"augmentation_panel_{safe_class_name}.png"
    plt.savefig(save_path)
    print(f"-> Saved augmentation panel to '{save_path}'")
    plt.close()


if __name__ == '__main__':
    # Using a dictionary for clearer, more organized input
    sample_images_dict = {
        "Non Demented": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data\Non Demented\OAS1_0001_MR1_mpr-1_118.jpg",
        "Very Mild Demented": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data\Very mild Dementia\OAS1_0003_MR1_mpr-1_148.jpg",
        "Demented": r"C:\Users\aravs\Desktop\Arav\Research\Divya_Maam\Data\Moderate Dementia\OAS1_0308_MR1_mpr-1_147.jpg"
    }

    # Check which of the provided paths are valid and generate a panel for each
    valid_images_dict = {name: path for name, path in sample_images_dict.items() if os.path.exists(path)}

    if not valid_images_dict:
        print("[Error] No valid image paths found. Please update the 'sample_images_dict' in the script.")
    else:
        print(f"Found {len(valid_images_dict)} valid image(s). Generating panels...")
        # MODIFIED: Loop through the dictionary and create one image per class
        for class_name, img_path in valid_images_dict.items():
            generate_sequential_augmentation_panel(class_name, img_path)