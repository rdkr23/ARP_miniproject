import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_feature_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=num_feature_classes + 1)

    return model


def train_step(model, optimizer, images, targets, device):
    """
    Performs one training step.

    Each image:
        Float tensor [3, H, W], preferably scaled to [0, 1].

    Each target:
        {
            "boxes": FloatTensor[N, 4],   # xmin, ymin, xmax, ymax
            "labels": LongTensor[N]       # feature classes start at 1
        }
    """
    model.train()

    images = [image.to(device) for image in images]
    targets = [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]

    loss_dict = model(images, targets)
    loss = sum(loss_dict.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.inference_mode()
def predict(model, images, device, score_threshold):
    """
    Returns detected boxes, labels and confidence scores.
    """
    model.eval()

    images = [image.to(device) for image in images]
    predictions = model(images)

    filtered_predictions = []

    for prediction in predictions:
        keep = prediction["scores"] >= score_threshold

        filtered_predictions.append({
            "boxes": prediction["boxes"][keep].cpu(),
            "labels": prediction["labels"][keep].cpu(),
            "scores": prediction["scores"][keep].cpu(),
        })

    return filtered_predictions