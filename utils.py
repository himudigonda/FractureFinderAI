
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# Non-Maximum Suppression (NMS)
def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break
        ious = np.array([iou(boxes[current], boxes[i]) for i in indices[1:]])
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)
    
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

# Data Loader Utility
def collate_fn(batch):
    return tuple(zip(*batch))

# Evaluation Metrics
def calculate_mAP(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection.
    Args:
    - pred_boxes (list of tensors): predicted bounding boxes and scores
    - true_boxes (list of tensors): ground truth bounding boxes
    - iou_threshold (float): IoU threshold to consider a prediction as true positive

    Returns:
    - mAP (float): mean Average Precision
    """
    # Code to calculate mAP
    pass

def calculate_mAUC(y_true, y_scores):
    """
    Calculate mean Area Under the Curve (mAUC) for classification.
    Args:
    - y_true (array-like): true labels
    - y_scores (array-like): predicted scores

    Returns:
    - mAUC (float): mean Area Under the Curve
    """
    return roc_auc_score(y_true, y_scores)

def calculate_mAP_score(y_true, y_scores):
    """
    Calculate mean Average Precision score for classification.
    Args:
    - y_true (array-like): true labels
    - y_scores (array-like): predicted scores

    Returns:
    - mAP_score (float): mean Average Precision score
    """
    return average_precision_score(y_true, y_scores)

# Save Model Weights
def save_model_weights(model, filepath):
    """
    Save the model weights to a file.
    Args:
    - model (torch.nn.Module): The model whose weights are to be saved
    - filepath (str): The file path where the weights should be saved
    """
    torch.save(model.state_dict(), filepath)

# Load Model Weights
def load_model_weights(model, filepath):
    """
    Load model weights from a file.
    Args:
    - model (torch.nn.Module): The model to load weights into
    - filepath (str): The file path from where the weights should be loaded
    """
    model.load_state_dict(torch.load(filepath))
