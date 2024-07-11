import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.optimize import linear_sum_assignment


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    if box1.size(0) == 0 or box2.size(0) == 0:
        return 0.0

    x1_max = torch.max(box1[:, None, 0], box2[:, 0])
    y1_max = torch.max(box1[:, None, 1], box2[:, 1])
    x2_min = torch.min(box1[:, None, 2], box2[:, 2])
    y2_min = torch.min(box1[:, None, 3], box2[:, 3])

    inter_area = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(
        y2_min - y1_max, min=0
    )
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area[:, None] + box2_area - inter_area

    iou = inter_area / union_area

    # Use Hungarian algorithm to find the best matching
    iou_matrix = iou.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    return iou[row_ind, col_ind].mean().item()


def calculate_dice(box1, box2):
    """
    Calculate the Dice score of two bounding boxes.
    """
    if box1.size(0) == 0 or box2.size(0) == 0:
        return 0.0

    x1_max = torch.max(box1[:, None, 0], box2[:, 0])
    y1_max = torch.max(box1[:, None, 1], box2[:, 1])
    x2_min = torch.min(box1[:, None, 2], box2[:, 2])
    y2_min = torch.min(box1[:, None, 3], box2[:, 3])

    inter_area = torch.clamp(x2_min - x1_max, min=0) * torch.clamp(
        y2_min - y1_max, min=0
    )
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    dice_score = (2 * inter_area) / (box1_area[:, None] + box2_area)

    # Use Hungarian algorithm to find the best matching
    dice_matrix = dice_score.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-dice_matrix)

    return dice_score[row_ind, col_ind].mean().item()


def visualize_image(image, annotations, categories):
    print("Visualizing image with annotations...")
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, label in zip(annotations["boxes"], annotations["labels"]):
        category = categories[label.item()]
        x, y, width, height = bbox.tolist()
        rect = patches.Rectangle(
            (x, y), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x, y, category, color="white", backgroundcolor="red")

    plt.axis("off")
    plt.show()
    print("Visualization completed")
