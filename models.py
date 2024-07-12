import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from utils import calculate_iou, calculate_dice, save_checkpoint, load_checkpoint
import logging

def get_model(num_classes):
    logging.info("Loading pre-trained Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logging.info("Model loaded and modified")
    return model

def train_model(model, dataloader, optimizer, device, num_epochs, checkpoint_path=None, scheduler=None):
    model.to(device)
    start_epoch = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer) + 1

    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        model.train()  # Ensure model is in training mode

        for batch_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        ):
            if len(images) == 0:
                continue

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
                    logging.info(
                        f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {losses.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}"
                    )
            model.train()  # Switch back to training mode

        logging.info(
            f"Epoch {epoch + 1} completed, Average Loss: {epoch_loss / len(dataloader):.4f}"
        )

        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch, checkpoint_path)

        if scheduler:
            scheduler.step(epoch_loss / len(dataloader))

    return model
