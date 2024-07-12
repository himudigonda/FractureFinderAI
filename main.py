import os
import torch
from torch.utils.data import DataLoader
from dataset import COCODataset
from models import get_model, train_model
import torchvision.transforms as T
import utils
from tqdm import tqdm
import yaml
import logging
from itertools import product
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))

def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Paths
    annotations_file = config['detection']['annotations_file']
    img_dir = config['detection']['img_dir']
    weights_path = config['detection']['weights_path']
    checkpoint_path = config['detection']['checkpoint_path']

    # Hyperparameters
    num_classes = config['detection']['num_classes']
    batch_size = config['detection']['batch_size']
    num_epochs = config['detection']['num_epochs']
    learning_rate = config['detection']['learning_rate']

    # Transforms
    train_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
    test_transform = T.Compose([T.ToTensor()])

    # Datasets
    logging.info("Loading training dataset...")
    train_dataset = COCODataset(annotations_file, img_dir, transform=train_transform)
    logging.info(f"Training dataset loaded with {len(train_dataset)} images")

    logging.info("Loading testing dataset...")
    test_dataset = COCODataset(annotations_file, img_dir, transform=test_transform)
    logging.info(f"Testing dataset loaded with {len(test_dataset)} images")

    # Dataloaders
    logging.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )
    logging.info("Data loaders created")

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Hyperparameter grid
    learning_rates = [0.001, 0.005, 0.01]
    weight_decays = [0.0001, 0.0005, 0.001]

    best_iou = 0
    best_dice = 0
    best_params = {}

    # Grid Search
    for lr, weight_decay in product(learning_rates, weight_decays):
        logging.info(f"Training with lr={lr}, weight_decay={weight_decay}")

        # Model
        model = get_model(num_classes)
        model.to(device)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = RAdam(params, lr=lr, weight_decay=weight_decay)

        # Scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

        # Train the model
        model = train_model(model, train_loader, optimizer, device, num_epochs, checkpoint_path, scheduler)

        # Evaluate the model
        model.eval()
        total_iou = 0
        total_dice = 0
        with torch.no_grad():
            for idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
                if len(images) == 0:
                    continue
                images = list(image.to(device) for image in images)
                outputs = model(images)
                for i, output in enumerate(outputs):
                    pred_boxes = output["boxes"].detach().cpu()
                    true_boxes = targets[i]["boxes"].detach().cpu()
                    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                        iou = 0.0
                        dice = 0.0
                    else:
                        iou = utils.calculate_iou(pred_boxes, true_boxes)
                        dice = utils.calculate_dice(pred_boxes, true_boxes)
                    total_iou += iou
                    total_dice += dice

        avg_iou = total_iou / len(test_loader)
        avg_dice = total_dice / len(test_loader)

        logging.info(f"lr={lr}, weight_decay={weight_decay}, Avg IoU={avg_iou:.4f}, Avg Dice={avg_dice:.4f}")

        scheduler.step(avg_iou)  # Use avg_iou for scheduling

        if avg_iou > best_iou and avg_dice > best_dice:
            best_iou = avg_iou
            best_dice = avg_dice
            best_params = {
                'learning_rate': lr,
                'weight_decay': weight_decay
            }
            # Save the best model
            torch.save(model.state_dict(), weights_path)
            logging.info(f"New best model saved with IoU={best_iou:.4f}, Dice={best_dice:.4f}")

    logging.info(f"Best parameters found: {best_params}, IoU={best_iou:.4f}, Dice={best_dice:.4f}")

if __name__ == "__main__":
    main()
