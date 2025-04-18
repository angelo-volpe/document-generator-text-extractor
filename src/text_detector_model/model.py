import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(num_classes):
    """
    Initialize a Faster R-CNN model with a pre-trained backbone
    and a custom number of output classes.
    """
    # Load a pre-trained model
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", trainable_backbone_layers=3)
    
    # Replace the pre-trained head with a new one for our task
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model