import csv
import json
import os
from PIL import Image  # Import Image from PIL


def csv_to_coco(csv_file, img_dir, output_file):
    # Initialize COCO format dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "fracture"},
            {"id": 2, "name": "old fracture"},
            {"id": 3, "name": "suspicious fracture"},
        ],
    }

    # Read the CSV file
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        image_id = 0
        annotation_id = 0
        image_map = {}

        for row in reader:
            image_file = row["path"]
            class_name = row["class"]
            x1, y1, x2, y2 = (
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            )

            if image_file not in image_map:
                # Add new image entry
                image_id += 1
                image_map[image_file] = image_id
                img_path = os.path.join(img_dir, image_file)
                width, height = Image.open(img_path).size
                coco_format["images"].append(
                    {
                        "id": image_id,
                        "file_name": image_file,
                        "width": width,
                        "height": height,
                    }
                )

            # Add annotation entry
            annotation_id += 1
            category_id = (
                1
                if class_name == "fracture"
                else 2 if class_name == "old fracture" else 3
            )
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_map[image_file],
                    "category_id": category_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO format JSON saved to {output_file}")


if __name__ == "__main__":
    csv_file = "data/data.csv"
    img_dir = "data"
    output_file = "data/data.json"
    csv_to_coco(csv_file, img_dir, output_file)
