# FractureFinderAI

This repository contains the code development for task 1 of FractureFinderAI.

## Environment Setup

1. **Create the environment:**
   ```bash
   mamba create -n endimension
   ```

2. **Activate the environment:**
   ```bash
   conda activate endimension
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Verify CUDA Availability

1. **Check NVIDIA GPU status:**
   ```bash
   nvidia-smi
   ```

2. **Verify CUDA availability in PyTorch:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Data Preparation

1. **Convert `data.csv` to COCO format `data.json`:**
   ```bash
   python cocoify.py
   ```

   This script will convert your `data.csv` file containing bounding box annotations to a COCO-style `data.json` file.

2. **Ensure your data directory structure:**

   ```
   data/
   ├── fracture_data/
   │   ├── <image1>.jpg
   │   ├── <image2>.jpg
   │   └── ...
   ├── non_fracture_data/
   │   ├── <image1>.jpg
   │   ├── <image2>.jpg
   │   └── ...
   ├── data.csv
   └── data.json
   ```

## Training and Evaluation

1. **Run the main script:**
   ```bash
   python main.py
   ```

   This script will train the model using the COCO-style `data.json` file and evaluate its performance. During training, it will also calculate and print the IoU and Dice score for each batch.

2. **Run the main bash file:**
   ```bash
   sh main.sh
   ```

## Visualizations

- **Training and Prediction Visualizations:**
  The script `main.py` includes visualizations of training images with their annotations and predictions. This helps in verifying the correctness of the dataset and the learning progress of the model.

## Troubleshooting and Debugging

- **Visualize Sample Training Images:**
  During the initial stages of the `main.py` script, sample training images along with their annotations are visualized to ensure the correctness of the dataset.

- **Visualize Predictions:**
  During training, predictions are visualized to help debug if the model is making reasonable predictions.

## File Descriptions

- `cocoify.py`: Script to convert `data.csv` to COCO format `data.json`.
- `dataset.py`: Dataset class to handle loading of images and annotations in COCO format.
- `models.py`: Model initialization, training loop, and metric calculations.
- `utils.py`: Utility functions for IoU and Dice score calculations and visualization functions.
- `main.py`: Main script to run training and evaluation of the model.
- `requirements.txt`: List of required packages for the project.

## Additional Information

- Ensure that the `data.csv` file and images are correctly placed in the `data` directory before running the scripts.
- The model uses a pre-trained Faster R-CNN with a ResNet50 backbone and is fine-tuned on the custom dataset.

By following the steps in this README, you can set up the environment, prepare the data, and train and evaluate the object detection model for fracture detection.