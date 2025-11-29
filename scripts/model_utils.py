## This script was consolidated using Google Gemini AI ##
import torch

def prepare_batch(images, targets, device, image_size=256, model_type='detr'):
    """
    Prepares a batch of images and targets for different object detection models.
    
    Args:
        images: Batch of images (N, C, H, W)
        targets: List of dictionaries containing 'boxes' and 'labels'
        device: 'cpu' or 'cuda'
        image_size: Input size of the images (default 256)
        model_type: 'detr', 'faster_rcnn', or 'yolo'
    """
    # --- 1. Image Processing (Universal) ---
    # Most pre-trained backbones (ResNet, etc.) expect 3-channel RGB
    pixel_values = torch.stack(images)
    if pixel_values.shape[1] == 1:
        pixel_values = pixel_values.repeat(1, 3, 1, 1)
    pixel_values = pixel_values.to(device)

    # --- 2. Target Processing (Model Specific) ---
    prepared_targets = []
    
    for t in targets:
        boxes = t['boxes'].to(device)
        labels = t['labels'].to(device)
        
        new_targets = {}

        if model_type == 'detr':
            # DETR requires: Normalized (cx, cy, w, h) in [0, 1]
            # DETR requires: Class labels starting at 0 (Background is usually last)
            if boxes.shape[0] > 0:
                # Convert xyxy to cxcywh
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                cx = boxes[:, 0] + w / 2
                cy = boxes[:, 1] + h / 2
                
                # Normalize
                new_boxes = torch.stack([cx/image_size, cy/image_size, w/image_size, h/image_size], dim=1)
            else:
                new_boxes = torch.zeros((0, 4), device=device)
            
            new_targets['boxes'] = new_boxes
            new_targets['class_labels'] = labels - 1 # 1-based (loader) -> 0-based (DETR)

        elif model_type == 'faster_rcnn':
            # Faster R-CNN requires: Absolute (x1, y1, x2, y2) pixels
            # Faster R-CNN requires: Class labels starting at 1 (0 is reserved for background)
            
            # Boxes are already xyxy from the loader, just move to device
            new_targets['boxes'] = boxes 
            new_targets['labels'] = labels # Keep 1-based labels

        elif model_type == 'yolo':
            # YOLO requires: Normalized (cx, cy, w, h) in [0, 1]
            # YOLO usually uses 0-based indexing
            
            # Same box logic as DETR
            if boxes.shape[0] > 0:
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]
                cx = boxes[:, 0] + w / 2
                cy = boxes[:, 1] + h / 2
                new_boxes = torch.stack([cx/image_size, cy/image_size, w/image_size, h/image_size], dim=1)
            else:
                new_boxes = torch.zeros((0, 4), device=device)
                
            new_targets['boxes'] = new_boxes
            new_targets['labels'] = labels - 1

        prepared_targets.append(new_targets)

    return pixel_values, prepared_targets