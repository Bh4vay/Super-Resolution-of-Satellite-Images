# Super-Resolution of Satellite Images

## Project Overview
This project focuses on generating high-resolution satellite imagery from low-resolution inputs using deep learning techniques. The primary objective is to enhance environmental monitoring and object detection.

## Dataset
- **Name**: SkyFusion: Aerial Object Detection
- **Base**: AiTOD v2 and Airbus Aircraft Detection
- **Classes**: Vehicles, Ships, Aircraft
- **Annotation Format**: MS-COCO (JSON)
- **Annotation Type**: Horizontal Bounding Boxes (HBB)
- **Structure**:
  - `train/annotations.coco.json`
  - `valid/annotations.coco.json`
  - `test/annotations.coco.json`

## Project Workflow
1. **Dataset Preparation**
   - Load and preprocess the dataset.
   - Split into training (20%), validation (5% of training set), and testing (10%) sets.

2. **Model Development**
   - Build a custom small object detection model from scratch without using pre-existing architectures like YOLO.
   - Implement a convolutional neural network (CNN) for classification.

3. **Training and Evaluation**
   - Train the model using the dataset.
   - Evaluate performance by calculating Mean Average Precision (mAP).

## Installation
### Requirements
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- JSON


## Model Architecture
- Conv2D layers for feature extraction
- Batch Normalization for stability
- Flatten and Dense layers for classification

## Evaluation
- The performance of the model is measured using the Mean Average Precision (mAP) metric.

## Acknowledgments
- Kaggle Dataset: Tiny Object Detection
- SkyFusion Dataset



