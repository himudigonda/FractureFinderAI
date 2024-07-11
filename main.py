import torch
from torch.utils.data import DataLoader
from dataset import COCODataset
from models import get_model, train_model
import torchvision.transforms as T
import utils
from tqdm import tqdm


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def main():
    # Paths
    annotations_file = "data/data.json"
    img_dir = "data/"
    weights_path = "weights/detection_model.pth"

    # Hyperparameters
    num_classes = (
        4  # 3 classes (fracture, old fracture, suspicious fracture) + background
    )
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.005

    # Transforms
    train_transform = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
    test_transform = T.Compose([T.ToTensor()])

    # Datasets
    print("Loading training dataset...")
    train_dataset = COCODataset(annotations_file, img_dir, transform=train_transform)
    print(f"Training dataset loaded with {len(train_dataset)} images")

    print("Loading testing dataset...")
    test_dataset = COCODataset(annotations_file, img_dir, transform=test_transform)
    print(f"Testing dataset loaded with {len(test_dataset)} images")

    # Dataloaders
    print("Creating data loaders...")
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
    print("Data loaders created")

    # Model
    print("Initializing model...")
    model = get_model(num_classes)
    print("Model initialized")

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Optimizer
    print("Creating optimizer...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )
    print("Optimizer created")

    # Train the model
    print("Starting training...")
    model = train_model(model, train_loader, optimizer, device, num_epochs)
    print("Training completed")

    # Save the model
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")

    # Evaluate the model
    print("Starting evaluation...")
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if len(images) == 0:
                continue
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                target = targets[i]
                output = output
                utils.visualize_image(img, output, train_dataset.categories)
                utils.visualize_image(img, target, train_dataset.categories)
    print("Evaluation completed")


if __name__ == "__main__":
    main()
