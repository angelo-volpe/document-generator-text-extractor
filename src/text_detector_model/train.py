import mlflow
import torch
from tqdm import tqdm
from text_detector_model.evaluation import evaluate_model


def train_model(
    model, train_data_loader, val_data_loader, optimizer, device="cpu", num_epochs=1
):
    """
    Train the Faster R-CNN model
    """

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        epoch_loss_dict = {}

        model.train()
        for images, targets in tqdm(train_data_loader):
            # Move images and targets to device
            images = [image.to(device) for image in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss_dict = {
                k: epoch_loss_dict.get(k, 0) + loss_dict.get(k, 0)
                for k in list(loss_dict)
            }

            # Backward pass
            losses.backward()
            optimizer.step()
            # if lr_scheduler is not None:
            #    lr_scheduler.step()

            # Update epoch loss
            epoch_loss += losses.item()

        epoch_loss_dict = {
            k: v.item() / len(train_data_loader) for k, v in epoch_loss_dict.items()
        }
        # Print epoch loss
        print(f"Train Loss: {epoch_loss/len(train_data_loader):.4f}")
        print(f"Train Loss Dict: {epoch_loss_dict}")
        mlflow.log_metric("train_loss", epoch_loss / len(train_data_loader), step=epoch)
        mlflow.log_metrics(epoch_loss_dict, step=epoch)

        # Evaluate COCO metrics
        [_, _], [precision, recall] = evaluate_model(
            model, val_data_loader, device, prediction_threshold=0.5, iou_thresold=0.95
        )
        mlflow.log_metric("precision", precision, step=epoch)
        mlflow.log_metric("recall", recall, step=epoch)
        print(f"Precision: {precision}, Recall: {recall}")

        mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")

    return model
