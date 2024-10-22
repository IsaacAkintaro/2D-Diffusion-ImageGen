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
            heapClassLabelImage = np.array(file['heapClassLabelImage'])

        heapClassLabelImage = np.where(np.isinf(heapClassLabelImage), 0, heapClassLabelImage)

        if heapClassLabelImage.ndim == 2:
            heapClassLabelImage = np.expand_dims(heapClassLabelImage, axis=0)

        if self.transform:
            heapRGBCompositeImage = self.transform(heapRGBCompositeImage)
            heapClassLabelImage = self.transform(heapClassLabelImage)
        else:
            heapRGBCompositeImage = torch.tensor(heapRGBCompositeImage).float()
            heapClassLabelImage = torch.tensor(heapClassLabelImage).long()

        combined_image = torch.cat((heapRGBCompositeImage, heapClassLabelImage.float()), dim=0)
        return combined_image

# Define the custom transformation class
# This class will be used to apply custom transformations to the data
# The __call__ method will apply the transformations to the input sample
# Reisizing of images due to mermory constraints
# The class will normalize the RGB images using the provided mean and standard deviation values
# The class label images will be resized using the nearest neighbour interpolation this is to ensure
# that the class labels don't get altered and retain their original values during the resizing process.
class ToTensorCustom:
    def __init__(self, mean_rgb, std_rgb, image_width, image_height):
        self.rgb_transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.Normalize(mean=mean_rgb, std=std_rgb),
        ])
        
        self.class_label_transform = transforms.Compose([
            transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __call__(self, sample):
        if sample.shape[0] == 3:  # RGB image
            return self.rgb_transform(torch.tensor(sample).float())
        else:  # Class label image
            return self.class_label_transform(torch.tensor(sample).long())  # Resize class labels with nearest neighbour interpolation

# Calculate the mean and standard deviation of the dataset for RGB images
def calculate_mean_std(dataset, batch_size):
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    mean_rgb = np.zeros(3)
    std_rgb = np.zeros(3)
    num_samples_rgb = np.zeros(3)

    for images in loader:
        rgb_images = images[:, :3, :, :].numpy()
        for i in range(3):
            non_zero_pixels = rgb_images[:, i, :, :][rgb_images[:, i, :, :] > 0.1]
            mean_rgb[i] += non_zero_pixels.sum()
            std_rgb[i] += (non_zero_pixels ** 2).sum()
            num_samples_rgb[i] += len(non_zero_pixels)
        
    mean_rgb /= num_samples_rgb
    std_rgb = np.sqrt(std_rgb / num_samples_rgb - mean_rgb ** 2)

    return mean_rgb, std_rgb

# Function to get the data loaders for training and validation
# This function will return the data loaders for training and validation
# The data will be loaded from the .mat files in the specified directory
# The batch size, image width, and image height are specified as input arguments
# The virtual size argument can be used to limit the number of samples loaded from the dataset
def get_class_data_loaders(mat_files_dir, batch_size, image_width, image_height, virtual_size=None):
    dataset = MatDataset(mat_files_dir, virtual_size=virtual_size)
    mean_rgb, std_rgb = calculate_mean_std(dataset, batch_size)
    
    transform = ToTensorCustom(mean_rgb, std_rgb, image_width, image_height)
    normalized_dataset = MatDataset(mat_files_dir, transform=transform, virtual_size=virtual_size)

    # Split the dataset into training and validation sets using 80:20 split
    train_size = int(0.8 * len(normalized_dataset))
    val_size = len(normalized_dataset) - train_size
    train_dataset, val_dataset = random_split(normalized_dataset, [train_size, val_size])

    # make sure to set num_workers=0 to avoid issues with the DataLoader on Windows or systems that make it difficult to do multiprocessing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader
