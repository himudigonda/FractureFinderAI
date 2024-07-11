import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from utils import calculate_iou, calculate_dice


def get_model(num_classes):
    print("Loading pre-trained Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("Model loaded and modified")
    return model


def train_model(model, dataloader, optimizer, device, num_epochs):
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        model.train()  # Ensure model is in training mode

        for batch_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        ):
            if len(images) == 0:
                continue

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Debug: print shapes of images and targets
            # print(f"Batch {batch_idx + 1}:")
            # print(f" - Images: {[image.shape for image in images]}")
            # print(f" - Targets: {targets}")

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # Temporarily switch to eval mode to calculate metrics
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                for i, output in enumerate(outputs):
                    pred_boxes = output["boxes"].detach().cpu()
                    true_boxes = targets[i]["boxes"].detach().cpu()
                    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                        iou = 0.0
                        dice = 0.0
                    else:
                        iou = calculate_iou(pred_boxes, true_boxes)
                        dice = calculate_dice(pred_boxes, true_boxes)
                    print(
                        f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {losses.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}"
                    )
            model.train()  # Switch back to training mode

        print(
            f"Epoch {epoch + 1} completed, Average Loss: {epoch_loss / len(dataloader):.4f}"
        )

    return model
