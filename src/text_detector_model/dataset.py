from torch.utils.data import Dataset
import json
from PIL import Image
import os
import torch
from torchvision import transforms
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
        image = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        img_annotations = self.annotations[img_filename]
        
        # Prepare bounding boxes and labels
        boxes = []
        labels = []
        
        for annotation in img_annotations:
            # Get bounding box coordinates
            coords = annotation["coords"]
            x1, y1, x2, y2 = coords[0][0], coords[0][1], coords[1][0], coords[2][1]
                
            # Add box coordinates
            boxes.append([x1, y1, x2, y2])
            
            # Add label
            field_type = annotation["name"]
            labels.append(self.classes[field_type])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target dictionary
        targets = {"boxes": boxes,
                   "labels": labels,}
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image, targets
