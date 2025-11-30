## This script was consolidated using Google Gemini AI ##
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import os

class FractureHDF5Dataset(Dataset):
    """
    A PyTorch Dataset to read the pre-processed HDF5 file.
    Can be initialized for 'classification' or 'detection' tasks.
    """
    def __init__(self, hdf5_path, transform=None, task='classification'):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hf = None 
        
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

        # Get Image
        image = self.hf['images'][idx] 
        # Convert to (1, H, W) tensor
        image = torch.from_numpy(image).float().unsqueeze(0) 
        
        # Apply Transforms (Augmentation) if provided
        if self.transform:
            image = self.transform(image)
        
        # Get Target
        if self.task == 'classification':
            label = self.hf['labels'][idx]
            label = torch.tensor(label, dtype=torch.long) 
            return image, label
            
        elif self.task == 'detection':
            bboxes_raw = self.hf['bboxes'][idx]
            bboxes_xywh = bboxes_raw[bboxes_raw[:, 0] != -1.0]
            bboxes_xywh_tensor = torch.from_numpy(bboxes_xywh).float()
            
            boxes_xyxy_tensor = torch.zeros_like(bboxes_xywh_tensor)
            if bboxes_xywh_tensor.shape[0] > 0:
                boxes_xyxy_tensor[:, 0] = bboxes_xywh_tensor[:, 0] 
                boxes_xyxy_tensor[:, 1] = bboxes_xywh_tensor[:, 1] 
                boxes_xyxy_tensor[:, 2] = bboxes_xywh_tensor[:, 0] + bboxes_xywh_tensor[:, 2] 
                boxes_xyxy_tensor[:, 3] = bboxes_xywh_tensor[:, 1] + bboxes_xywh_tensor[:, 3] 

            num_boxes = boxes_xyxy_tensor.shape[0]
            box_labels = torch.ones((num_boxes,), dtype=torch.int64) 
            
            target = {
                "boxes": boxes_xyxy_tensor,
                "labels": box_labels
            }
            
            return image, target

def get_dataloaders(hdf5_path, batch_size, task='classification', num_workers=2, train_transform=None, val_transform=None):
    """
    Creates DataLoaders. Accepts separate transforms for train and val.
    """
    print(f"Loading dataset for task: '{task}'...")
    
    # 1. Initialize two separate dataset objects
    # One with augmentation (train), one without (val/test)
    train_base = FractureHDF5Dataset(hdf5_path, task=task, transform=train_transform)
    val_base = FractureHDF5Dataset(hdf5_path, task=task, transform=val_transform)
    
    print("Loading splits...")
    with h5py.File(hdf5_path, 'r') as f:
        splits = f['split'][:] 

    # 2. Create Subsets mapping to the correct base dataset
    train_dataset = Subset(train_base, np.where(splits == 0)[0])
    val_dataset = Subset(val_base, np.where(splits == 1)[0])
    test_dataset = Subset(val_base, np.where(splits == 2)[0])

    # 3. Config
    def detection_collate_fn(batch):
        return tuple(zip(*batch))
    
    collate_fn = detection_collate_fn if task == 'detection' else None
    
    # Safety check for workers
    is_persistent = (num_workers > 0)

    # 4. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=is_persistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=is_persistent
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=is_persistent
    )

    print("DataLoaders created.")
    return train_loader, val_loader, test_loader