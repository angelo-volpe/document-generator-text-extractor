import torch


def evaluate_model(model, data_loader, device):
    """
    Evaluate the Faster R-CNN model
    """
    # Set model to evaluation mode
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Move images to device
            images = [image.to(device) for image in images]
            
            # Get predictions
            predictions = model(images)
            
            # Process each prediction
            for i, prediction in enumerate(predictions):
                target = targets[i]
                image_id = target['image_id'].item()
                
                # Get predicted boxes, labels, and scores
                pred_boxes = prediction['boxes'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()
                pred_scores = prediction['scores'].cpu().numpy()
                
                # Get ground truth boxes and labels
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                results.append({
                    'image_id': image_id,
                    'pred_boxes': pred_boxes,
                    'pred_labels': pred_labels,
                    'pred_scores': pred_scores,
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels
                })
    
    return results