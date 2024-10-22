import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Define the dataset class
# This class will be used to load the data from the .mat files
# The __getitem__ method will return a sample from the dataset
# The __len__ method will return the number of samples in the dataset

class MatDataset(Dataset):
    def __init__(self, mat_files_dir, transform=None, virtual_size=None):
        self.mat_files_dir = mat_files_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(mat_files_dir) if f.endswith('.mat')]
        self.virtual_size = virtual_size

    def __len__(self):
        return self.virtual_size if self.virtual_size else len(self.file_names)

    def __getitem__(self, idx):
        if self.virtual_size:
            idx = np.random.randint(len(self.file_names))
        mat_file_path = os.path.join(self.mat_files_dir, self.file_names[idx])
        with h5py.File(mat_file_path, 'r') as file:
            heapRGBCompositeImage = np.array(file['heapRGBCompositeImage'])
            heapDepthImage = np.array(file['heapDepthImage'])

        heapDepthImage = np.where(np.isinf(heapDepthImage), 0, heapDepthImage)
        heapDepthImage = np.expand_dims(heapDepthImage, axis=0) if heapDepthImage.shape[0] != 1 else heapDepthImage

        if self.transform:
            heapRGBCompositeImage = self.transform(heapRGBCompositeImage)
            heapDepthImage = self.transform(heapDepthImage)
        else:
            heapRGBCompositeImage = torch.tensor(heapRGBCompositeImage).float()
            heapDepthImage = torch.tensor(heapDepthImage).float()

        combined_image = torch.cat((heapRGBCompositeImage, heapDepthImage), dim=0)
        return combined_image

# Define the custom transformation class
# This class will be used to apply custom transformations to the data
# The __call__ method will apply the transformations to the input sample
# Reisizing of images due to mermory constraints
# The class will normalize the RGB and depth images using the provided mean and standard deviation values

class ToTensorCustom:
    def __init__(self, mean_rgb, std_rgb, mean_depth, std_depth, image_width, image_height):
        self.rgb_transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.Normalize(mean=mean_rgb, std=std_rgb),
        ])
        self.depth_transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.Normalize(mean=[mean_depth], std=[std_depth]),
        ])

    def __call__(self, sample):
        if sample.shape[0] == 3:
            return self.rgb_transform(torch.tensor(sample).float())
        elif sample.shape[0] == 1:
            return self.depth_transform(torch.tensor(sample).float())
        else:
            raise ValueError("Unexpected shape of the sample. Expected 3 (RGB) or 1 (Depth) channel.")
        
# Function to calculate the mean and standard deviation of the dataset
# RGB has three channels, while depth has one channel, so will need the mean and standard deviation for each channel
def calculate_mean_std(dataset, batch_size):
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    mean_rgb = np.zeros(3)
    mean_depth = 0
    std_rgb = np.zeros(3)
    std_depth = 0
    num_samples_rgb = np.zeros(3)
    num_samples_depth = 0

    for images in loader:
        rgb_images = images[:, :3, :, :].numpy()
        depth_images = images[:, -1, :, :].numpy()
        depth_images = np.where(np.isinf(depth_images), 0, depth_images)

        for i in range(3):
            non_zero_pixels = rgb_images[:, i, :, :][rgb_images[:, i, :, :] > 0.1]
            mean_rgb[i] += non_zero_pixels.sum()
            std_rgb[i] += (non_zero_pixels ** 2).sum()
            num_samples_rgb[i] += len(non_zero_pixels)
        # Calculate mean and std for depth images
        # Only consider non-zero depth pixels, as zero values are invalid, which in the .mat files are represented as infinity i.e air
        non_zero_depth_pixels = depth_images[depth_images > 0.1]
        mean_depth += non_zero_depth_pixels.sum()
        std_depth += (non_zero_depth_pixels ** 2).sum()
        num_samples_depth += len(non_zero_depth_pixels)

    mean_rgb /= num_samples_rgb
    std_rgb = np.sqrt(std_rgb / num_samples_rgb - mean_rgb ** 2)
    mean_depth /= num_samples_depth
    std_depth = np.sqrt(std_depth / num_samples_depth - mean_depth ** 2)

    return mean_rgb, std_rgb, mean_depth, std_depth

# Function to get the data loaders for the training and validation sets
# The function will return the training and validation data loaders, this can now be used to generate using validation data as a form of conditioning
# The data will be loaded from the .mat files in the specified directory
# The batch size, image width, and image height are specified as input arguments
# The virtual size argument can be used to limit the number of samples loaded from the dataset
def get_depth_data_loaders(mat_files_dir, batch_size, image_width, image_height, virtual_size=None):
    dataset = MatDataset(mat_files_dir, virtual_size=virtual_size)
    mean_rgb, std_rgb, mean_depth, std_depth = calculate_mean_std(dataset, batch_size)
    
    transform = ToTensorCustom(mean_rgb, std_rgb, mean_depth, std_depth, image_width, image_height)
    normalized_dataset = MatDataset(mat_files_dir, transform=transform, virtual_size=virtual_size)

    # Split the dataset into training and validation sets using 80:20 split
    train_size = int(0.8 * len(normalized_dataset))
    val_size = len(normalized_dataset) - train_size
    train_dataset, val_dataset = random_split(normalized_dataset, [train_size, val_size])

    # make sure to set num_workers=0 to avoid issues with the DataLoader on Windows or systems that make it difficult to do multiprocessing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader
