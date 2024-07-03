import yaml
import torch
from torch.utils.data import DataLoader
from dataset import get_classification_dataloader, get_localization_dataloader, ChestXRayLocalizationDataset
from classification_models import ResNetClassifier
from localization_models import get_faster_rcnn_model
from utils import save_model_weights, load_model_weights, calculate_mAUC
import pandas as pd

def train_classification_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    save_model_weights(model, 'weights/classification_model_weights.pth')

def filter_images_with_fractures(classification_model, dataloader, device):
    classification_model.to(device)
    classification_model.eval()
    fracture_images = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = classification_model(images)
            predicted = (outputs > 0.5).cpu().numpy()
            fracture_images.extend(images[predicted[:, 0] == 1].cpu().numpy())
    return fracture_images

def create_localization_dataframe(fracture_images, original_csv):
    original_df = pd.read_csv(original_csv)
    new_df = original_df[original_df['image_path'].isin(fracture_images)]
    new_csv_path = 'data/fracture_images.csv'
    new_df.to_csv(new_csv_path, index=False)
    return new_csv_path

def train_localization_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    save_model_weights(model, 'weights/localization_model_weights.pth')

def main():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Classification Data Loaders
    classification_train_loader = get_classification_dataloader(
        config['classification']['csv_file'], 
        config['classification']['root_dir'], 
        batch_size=config['classification']['batch_size'], 
        train=True
    )
    classification_test_loader = get_classification_dataloader(
        config['classification']['csv_file'], 
        config['classification']['root_dir'], 
        batch_size=config['classification']['batch_size'], 
        train=False
    )

    # Train Classification Model
    classification_model = ResNetClassifier(
        num_classes=config['classification']['num_classes'], 
        pretrained=config['classification']['pretrained'], 
        weights_path=config['classification']['weights_path']
    )
    criterion_classification = torch.nn.BCELoss()
    optimizer_classification = torch.optim.Adam(classification_model.parameters(), lr=0.001)
    train_classification_model(classification_model, classification_train_loader, criterion_classification, optimizer_classification, num_epochs=10, device=device)

    # Filter Images with Fractures
    fracture_images = filter_images_with_fractures(classification_model, classification_test_loader, device)
    
    # Create new CSV for localization training
    new_csv_path = create_localization_dataframe(fracture_images, config['localization']['csv_file'])

    # Localization Data Loaders
    localization_train_loader = get_localization_dataloader(
        new_csv_path, 
        config['localization']['root_dir'], 
        batch_size=config['localization']['batch_size'], 
        train=True
    )
    localization_test_loader = get_localization_dataloader(
        new_csv_path, 
        config['localization']['root_dir'], 
        batch_size=config['localization']['batch_size'], 
        train=False
    )

    # Train Localization Model
    localization_model = get_faster_rcnn_model(
        num_classes=config['localization']['num_classes'], 
        weights_path=config['localization']['weights_path']
    )
    criterion_localization = None  # Localization models have their own loss functions
    optimizer_localization = torch.optim.Adam(localization_model.parameters(), lr=0.001)
    train_localization_model(localization_model, localization_train_loader, criterion_localization, optimizer_localization, num_epochs=10, device=device)

if __name__ == '__main__':
    main()
