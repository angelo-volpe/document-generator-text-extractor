from torch.utils.data import Dataset
import json
import os
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from pathlib import Path


# Custom dataset for document images with bounding box annotations
class DocumentDataset(Dataset):
    def __init__(self, images_dir: Path, annotation_file: Path, transform=None):
        """
        Args:
            images_dir: Directory containing document images
            annotation_file: JSON containing bounding box annotations
            transform: Optional image transformations
        """
        self.images_dir = images_dir
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.transform = transform
        
        # Group annotations by filename
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        self.classes = {field: i+1 for i, field in enumerate(map(lambda x: x["name"], 
                                                                 list(self.annotations.values())[0]))}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image
        img_path = self.images_dir / self.images[idx]
        img_filename = img_path.name
        image = read_image(img_path)
        image = tv_tensors.Image(image)

        # Get annotations for this image
        img_annotations = self.annotations[img_filename]
        
        # Prepare bounding boxes and labels
        boxes = []
        labels = []
        
        for annotation in img_annotations:
            # Get bounding box coordinates
            coords = annotation["coords"]
            x1, y1, x2, y2 = (coords[0][0], coords[0][1], coords[2][0], coords[2][1])
                
            # Add box coordinates
            boxes.append([x1, y1, x2, y2])
            
            # Add label
            field_type = annotation["name"]
            labels.append(self.classes[field_type])
        
        # Convert to tensors
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target dictionary
        targets = {"image_id": idx,
                   "boxes": boxes,
                   "labels": labels}
        
        # Apply transformations if provided
        if self.transform:
            image, targets = self.transform(image, targets)

        return image, targets
