## This script was made using Google Gemini AI ##
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np

class FractureHDF5Dataset(Dataset):
    """
    A PyTorch Dataset to read the pre-processed HDF5 file.
    """
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hf = None # File handle, left open for speed
        
        # Open the file once just to get the total length
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                self.length = len(f['labels'])
        except FileNotFoundError:
            print(f"ERROR: HDF5 file not found at {hdf5_path}")
            raise
        except Exception as e:
            print(f"Error opening HDF5 file: {e}")
            raise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open the file here if it's not open
        if self.hf is None:
            self.hf = h5py.File(self.hdf5_path, 'r')

        # --- Get Image ---
        # (H, W) -> (1, H, W) for a 1-channel grayscale image
        image = self.hf['images'][idx]
        image = torch.from_numpy(image).float().unsqueeze(0) # (1, 256, 256)
        
        # --- Get Label (for CLASSIFICATION) ---
        label = self.hf['labels'][idx]
        label = torch.tensor(label, dtype=torch.long) # 0 for no fracture, 1 for fracture

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(hdf5_path, batch_size, num_workers=2):
    """
    Creates and returns the pre-split train, val, and test DataLoaders.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
        batch_size (int): Batch size for the DataLoaders.
        num_workers (int): Number of workers for the DataLoaders.
    
    Returns:
        (DataLoader, DataLoader, DataLoader): train_loader, val_loader, test_loader
    """
    
    print("Loading dataset...")
    # Create the full dataset instance
    full_dataset = FractureHDF5Dataset(hdf5_path)
    
    print("Loading pre-defined train/val/test splits...")
    # Load the pre-defined split assignments from the file
    with h5py.File(hdf5_path, 'r') as f:
        splits = f['split'][:] # [:] loads it into memory

    # Get the indices for each split
    train_indices = np.where(splits == 0)[0]
    val_indices = np.where(splits == 1)[0]
    test_indices = np.where(splits == 2)[0]

    print(f"Total samples: {len(splits)}")
    print(f"  Training indices:   {len(train_indices)}")
    print(f"  Validation indices: {len(val_indices)}")
    print(f"  Test indices:       {len(test_indices)}")

    # Create PyTorch Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle the training set
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle val
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test
        num_workers=num_workers,
        pin_memory=True
    )

    print("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader