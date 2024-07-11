import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class COCODataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file) as f:
            self.coco = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in self.coco["categories"]}

        # Create an index for annotations to speed up data loading
        self.annotation_index = self.create_annotation_index()

    def create_annotation_index(self):
        index = {}
        for annotation in self.annotations:
            img_id = annotation["image_id"]
            if img_id not in index:
                index[img_id] = []
            index[img_id].append(annotation)
        return index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        if img_id in self.annotation_index:
            for annotation in self.annotation_index[img_id]:
                x_min, y_min, width, height = annotation["bbox"]
                if width > 0 and height > 0:
                    boxes.append([x_min, y_min, x_min + width, y_min + height])
                    labels.append(annotation["category_id"])

        if not boxes:
            return None  # Skip negative images

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            img = self.transform(img)

        return img, target
