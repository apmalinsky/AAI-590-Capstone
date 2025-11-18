## This script was made using Google Gemini AI ##
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np

class FractureHDF5Dataset(Dataset):
    """
    A PyTorch Dataset to read the pre-processed HDF5 file.
    Can be initialized for 'classification' or 'detection' tasks.
    """
    def __init__(self, hdf5_path, transform=None, task='classification'):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hf = None # File handle, left open for speed
        
        if task not in ['classification', 'detection']:
            raise ValueError("Task must be 'classification' or 'detection'")
        self.task = task
        
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
        if self.hf is None:
            self.hf = h5py.File(self.hdf5_path, 'r')

        # --- Get Image (Same for both tasks) ---
        image = self.hf['images'][idx]
        image = torch.from_numpy(image).float().unsqueeze(0) # (1, 256, 256)
        
        if self.transform:
            image = self.transform(image)
        
        # --- Get Target (Depends on the task) ---
        
        if self.task == 'classification':
            # Task 1: Return (image, label)
            label = self.hf['labels'][idx]
            label = torch.tensor(label, dtype=torch.long) # 0 or 1
            return image, label
            
        elif self.task == 'detection':
            # Task 2: Return (image, target_dict)
            
            # 1. Get the bounding boxes
            bboxes_raw = self.hf['bboxes'][idx]
            
            # 2. Filter out the -1 padding
            bboxes = bboxes_raw[bboxes_raw[:, 0] != -1.0]
            
            # 3. Convert to a tensor
            bboxes_tensor = torch.from_numpy(bboxes).float()
            
            # 4. Handle edge case: no boxes (for a positive sample? should be rare)
            if bboxes_tensor.shape[0] == 0:
                # No boxes, return an empty target
                bboxes_tensor = torch.empty((0, 4), dtype=torch.float32)
                
            # 5. Create class labels for each box
            # Since we only have one class ("fracture"), they are all label '1'
            # (Label '0' is reserved for the background)
            num_boxes = bboxes_tensor.shape[0]
            box_labels = torch.ones((num_boxes,), dtype=torch.int64)
            
            # 6. Create the final target dictionary
            target = {
                "boxes": bboxes_tensor,
                "labels": box_labels
            }
            
            return image, target

def get_dataloaders(hdf5_path, batch_size, task='classification', num_workers=2):
    """
    Creates and returns the pre-split train, val, and test DataLoaders.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
        batch_size (int): Batch size for the DataLoaders.
        task (str): 'classification' or 'detection'. This determines
                    what the dataloader will yield.
        num_workers (int): Number of workers for the DataLoaders.
    
    Returns:
        (DataLoader, DataLoader, DataLoader): train_loader, val_loader, test_loader
    """
    
    print(f"Loading dataset for task: '{task}'...")
    full_dataset = FractureHDF5Dataset(hdf5_path, task=task)
    
    print("Loading pre-defined train/val/test splits...")
    with h5py.File(hdf5_path, 'r') as f:
        splits = f['split'][:] 

    train_indices = np.where(splits == 0)[0]
    val_indices = np.where(splits == 1)[0]
    test_indices = np.where(splits == 2)[0]

    print(f"Total samples: {len(splits)}")
    print(f"  Training indices:   {len(train_indices)}")
    print(f"  Validation indices: {len(val_indices)}")
    print(f"  Test indices:       {len(test_indices)}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # --- Define a collate_fn for detection ---
    # This is needed because 'target' dicts can't be auto-batched
    def detection_collate_fn(batch):
        return tuple(zip(*batch))
    
    collate_function = detection_collate_fn if task == 'detection' else None

    # Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_function # Use custom collate for detection
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_function
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_function
    )

    print("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader