import argparse
from dataset import DocumentDataset
from train import train_model
from model import get_faster_rcnn_model
from evaluation import evaluate_model
from pathlib import Path
from torchvision.transforms import v2 as T
import mlflow
import torch
import os
from torch.utils.data import DataLoader
from logging_config import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base_path", type=str, required=True)
    parser.add_argument("--document_id", type=str, required=True)
    parser.add_argument("--data_version", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--num_epochs", type=int, required=False, default=10)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001)
    parser.add_argument("--momentum", type=float, required=False, default=0.9)
    parser.add_argument("--weight_decay", type=float, required=False, default=0.0005)

    args = parser.parse_args()

    logger.info("Starting training script")
    logger.info(f"Arguments: {args}")

    # Set the tracking URI for MLflow
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(f"text_detector_model_{args.document_id}")

    # Define transformations for input images
    transform = T.Compose([
        T.ToDtype(torch.float, scale=True),
        T.Resize((800, 600)),
        T.ToPureTensor()
    ])

    # Load data
    documents_dir = Path(args.input_base_path) / f"document_{args.document_id}" / args.data_version

    train_dataset = DocumentDataset(images_dir=documents_dir / "train", annotation_file=documents_dir / "train_labels.json", transform=transform)
    test_val_dataset = DocumentDataset(images_dir=documents_dir / "test", annotation_file=documents_dir / "test_labels.json", transform=transform)

    # Split into val/test
    val_size = int(0.5 * len(test_val_dataset))
    test_size = len(test_val_dataset) - val_size

    val_dataset, test_dataset = torch.utils.data.random_split(
        test_val_dataset, [val_size, test_size]
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Initialize model
    model = get_faster_rcnn_model(len(train_dataset.classes) + 1)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train the model
    logger.info("Starting training")
    with mlflow.start_run():
        params = {
            "num_epochs": args.num_epochs,
            "optimizer": optimizer.__class__.__name__,
            "batch_size": args.batch_size,
            "data_version": args.data_version,
            "learning_rate": args.learning_rate,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay
        }
        mlflow.log_params(params)

        model = train_model(model=model, 
                            train_data_loader=train_loader, 
                            val_data_loader=val_loader, 
                            optimizer=optimizer, 
                            device=args.device, 
                            num_epochs=args.num_epochs)

    # Evaluate the model
    logger.info("Evaluating model on test dataset")
    [_, _], [precision, recall] = evaluate_model(model, val_loader, args.device, prediction_threshold=0.5, iou_thresold=0.90)

    logger.info(f"Precision: {precision}, Recall: {recall}")