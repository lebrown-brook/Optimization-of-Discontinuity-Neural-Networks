import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.io as sio
import os

class SingleImageDataset(Dataset):
    def __init__(self, image_path, mat_path=None, num_samples=None, sampling_strategy='uniform'):
        """
        PyTorch Dataset for single-image coordinate-to-color mapping (Implicit Neural Representation).
        
        Args:
            image_path (str): Path to the image file (e.g. .jpg)
            mat_path (str, optional): Path to the corresponding .mat edge data file.
            num_samples (int, optional): Number of pixels to sample for training. 
                                         If None, uses all pixels.
            sampling_strategy (str): 'uniform' or 'gaussian'
        """
        super().__init__()
        self.image_path = image_path
        self.mat_path = mat_path
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        self.W, self.H = img.size
        
        # Convert to tensor and normalize to [0, 1]
        # img_tensor shape: [H, W, 3]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        
        # Generate coordinate grid
        # y goes from 0 to H-1, x goes from 0 to W-1
        # meshgrid 'ij' means first dim is y, second is x
        ys, xs = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        
        # Flatten to [N, 2] and [N, 3]
        self.coords = torch.stack([xs.flatten(), ys.flatten()], dim=1).float() # [N, 2]
        self.colors = img_tensor.view(-1, 3) # [N, 3]
        
        # Normalize coordinates to [-1, 1]
        # x_norm = (x / (W-1)) * 2 - 1
        # y_norm = (y / (H-1)) * 2 - 1
        self.coords[:, 0] = (self.coords[:, 0] / (self.W - 1)) * 2.0 - 1.0
        self.coords[:, 1] = (self.coords[:, 1] / (self.H - 1)) * 2.0 - 1.0
        
        # Perform sampling if requested
        if num_samples is not None and num_samples < self.coords.shape[0]:
            self._sample_data(num_samples, sampling_strategy)
            
        # Optional: Load ground truth
        self.edges = None
        if mat_path and os.path.exists(mat_path):
            self.edges = self.load_ground_truth(mat_path)
            
    def _sample_data(self, num_samples, strategy):
        total_pixels = self.coords.shape[0]
        
        if strategy == 'uniform':
            indices = torch.randperm(total_pixels)[:num_samples]
            
        elif strategy == 'gaussian':
            # Sample more densely around the center
            # Generate gaussian weights over x and y [-1, 1]
            sigma = 0.5
            dist = torch.exp(-(self.coords[:, 0]**2 + self.coords[:, 1]**2) / (2 * sigma**2))
            weights = dist / dist.sum()
            indices = torch.multinomial(weights, num_samples, replacement=False)
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        self.coords = self.coords[indices]
        self.colors = self.colors[indices]

    def load_ground_truth(self, mat_path):
        """
        Placeholder function to read the .mat segmentation data.
        EdgeDetectionDataset typically provides cell arrays in the .mat file.
        """
        try:
            mat = sio.loadmat(mat_path)
            # TODO: Extract boundary maps (e.g., groundTruth{i}.Boundaries)
            return mat
        except Exception as e:
            print(f"Failed to load {mat_path}: {e}")
            return None

    def full_grid(self):
        """Return coords for every pixel (no sampling), normalized to [-1, 1], plus (H, W)."""
        ys, xs = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing='ij')
        coords = torch.stack([xs.flatten(), ys.flatten()], dim=1).float()
        coords[:, 0] = (coords[:, 0] / (self.W - 1)) * 2.0 - 1.0
        coords[:, 1] = (coords[:, 1] / (self.H - 1)) * 2.0 - 1.0
        return coords, self.H, self.W

    def full_luminance(self):
        """Return the full H×W ground-truth luminance array in [0, 1] (no sampling)."""
        img = Image.open(self.image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
        return 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        # Return [2] (x,y) and [3] (r,g,b)
        return self.coords[idx], self.colors[idx]
