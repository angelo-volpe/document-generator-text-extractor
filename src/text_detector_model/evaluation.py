import torch
from tqdm import tqdm
import pandas as pd
from shapely import Polygon, Point


def box_to_poly(box):
    x1, y1, x2, y2 = box
    return Polygon([Point(x1, y1), Point(x1, y2), Point(x2, y2), Point(x2, y1)])


def evaluate_model(model, data_loader, device, prediction_threshold=0.5, iou_thresold=0.95):
    print("Evaluating model")
    model = model.to(device)
    model.eval()
    
    # run predictions
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # calculate metrics
    false_negatives_df = pd.DataFrame()
    positives_df = pd.DataFrame()
    for prediction, pred_targets in zip(all_predictions, all_targets):
        prediction_mask = prediction["scores"] > prediction_threshold
        prediction["boxes"] = prediction["boxes"][prediction_mask]
        prediction["labels"] = prediction["labels"][prediction_mask]
        prediction["scores"] = prediction["scores"][prediction_mask]

        prediction_results = pd.DataFrame(zip(prediction["scores"].cpu().numpy(), 
                                                prediction["boxes"].cpu().numpy(), 
                                                prediction["labels"].cpu().numpy()), 
                                                columns=["score", "predicted_box", "class"])
        # if multiple predictions for one class take the one with highest score, 
        # knowing that we have only one item per class by definition
        prediction_results = prediction_results.sort_values(by="class", ascending=False).drop_duplicates(subset=["class"])

        targets_df = pd.DataFrame(zip(pred_targets["boxes"].cpu().numpy(), 
                                      pred_targets["labels"].cpu().numpy()), 
                                      columns=["target_box", "class"])
        targets_df["image_id"] = pred_targets["image_id"]
        prediction_results = targets_df.merge(prediction_results, on="class", how="left")

        false_negatives_df = pd.concat([false_negatives_df, prediction_results[prediction_results["score"].isnull()]])
        
        positives = prediction_results[~prediction_results["score"].isnull()].copy()

        if len(positives) == 0:
            continue

        # convert to shapely polygons to facilitate geometrical calculations
        positives["predicted_poly"] = positives["predicted_box"].apply(box_to_poly)
        positives["target_poly"] = positives["target_box"].apply(box_to_poly)

        # calculate Intersect over Union
        positives["intersection_area"] = positives[["target_poly", "predicted_poly"]] \
            .apply(lambda x: x["target_poly"].intersection(x["predicted_poly"]).area, axis=1)
        positives["union_area"] = positives[["target_poly", "predicted_poly"]] \
            .apply(lambda x: x["target_poly"].union(x["predicted_poly"]).area, axis=1)
        positives["IoU"] = positives["intersection_area"] / positives["union_area"]

        positives_df = pd.concat([positives_df, positives])

    # calculate precision and recall
    false_negatives = len(false_negatives_df)
    if len(positives_df) > 0:
        true_positives = len(positives_df[positives_df["IoU"] >= iou_thresold])
        false_positives = len(positives_df[positives_df["IoU"] < iou_thresold])
        precision = true_positives / (true_positives + false_positives)
    else:
        true_positives = 0
        false_positives = 0
        precision = 0

    recall = true_positives / (true_positives + false_negatives)

    return [false_negatives_df, positives_df], [precision, recall]