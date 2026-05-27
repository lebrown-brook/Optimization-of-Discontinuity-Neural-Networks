import torch
from image_dataset import SingleImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def test_dataloader():
    # Attempt to locate one of the training images
    dataset_dir = os.path.join("EdgeDetectionData", "images", "train")
    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} not found. Please verify your data path.")
        return
        
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
    if not image_files:
        print("No .jpg files found.")
        return
        
    img_name = image_files[0]
    img_path = os.path.join(dataset_dir, img_name)
    mat_path = os.path.join("EdgeDetectionData", "ground_truth", "train", img_name.replace('.jpg', '.mat'))
    
    print(f"Testing with image: {img_path}")
    
    # Test Uniform Sampling
    num_samples = 15000
    dataset_uniform = SingleImageDataset(img_path, mat_path=mat_path, num_samples=num_samples, sampling_strategy='uniform')
    
    # Test Gaussian Sampling
    dataset_gaussian = SingleImageDataset(img_path, mat_path=mat_path, num_samples=num_samples, sampling_strategy='gaussian')
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for ax, ds, title in zip(axes, [dataset_uniform, dataset_gaussian], ['Uniform Sampling', 'Gaussian Sampling']):
        coords = ds.coords.numpy()
        colors = ds.colors.numpy()
        
        # Plot scattered points. We flip Y axis so the image isn't upside down.
        ax.scatter(coords[:, 0], -coords[:, 1], c=colors, s=1.5)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
    plt.tight_layout()
    plt.savefig('dataloader_test.png')
    print("Saved 'dataloader_test.png' with visualizations of sampled points.")
    
if __name__ == "__main__":
    import torchvision  # Verifying imports
    test_dataloader()
