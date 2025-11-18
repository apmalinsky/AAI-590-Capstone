## This script was made using Google Gemini AI ##
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
import numpy as np
import random
import os

def visualize_samples_with_boxes(hdf5_path, num_samples=5):
    """
    Plots random samples that HAVE fractures, along with their
    bounding boxes, directly from the HDF5 file.
    """
    print(f"Loading samples from {hdf5_path}...")
    
    with h5py.File(hdf5_path, 'r') as hf:
        # 1. Load all labels and find the ones with fractures
        all_labels = hf['labels'][:]
        positive_indices = np.where(all_labels == 1)[0]
        
        if len(positive_indices) == 0:
            print("Error: No positive samples found in the dataset.")
            return
            
        print(f"Found {len(positive_indices)} positive samples.")

        # 2. Pick N random samples from the positive-only list
        if num_samples > len(positive_indices):
            print(f"Warning: Requested {num_samples} but only {len(positive_indices)} are available.")
            num_samples = len(positive_indices)
            
        sample_indices = random.sample(list(positive_indices), num_samples)
        
        # 3. Create a plot for each sample
        # Make the plot wider to see details
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 10))
        if num_samples == 1:
            axes = [axes] # Make it iterable if there's only one

        for ax, idx in zip(axes, sample_indices):
            # Load the data for this one sample
            image = hf['images'][idx]
            bboxes = hf['bboxes'][idx]
            uid = hf['StudyInstanceUID'][idx]
            slice_num = hf['SliceNumber'][idx]
            
            # Filter out the "-1" padding on the bounding boxes
            bboxes = [box for box in bboxes if box[0] != -1.0]
            
            # Plot the image
            ax.imshow(image, cmap='bone')
            ax.set_title(f"UID: {uid}\nSlice: {slice_num} (Index: {idx})")
            
            # Plot each bounding box
            for box in bboxes:
                # bboxes are in [x, y, w, h] format
                x, y, w, h = box
                
                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none' # Transparent face
                )
                
                # Add the patch to the Axes
                ax.add_patch(rect)
                
        plt.tight_layout()
        plt.show()