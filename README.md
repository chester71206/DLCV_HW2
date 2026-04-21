# NYCU Computer Vision 2026 HW2

- Student ID: 314551178
- Name: 陳鎮成

## Introduction
This repository contains the source code for Homework 2: Digit Detection. The model uses a DETR-based architecture (RT-DETRv2) with a ResNet-50 backbone to detect digits (0-9) from RGB images.

## Environment Setup
It is recommended to use Python 3.9 or higher. To install the required dependencies, please run:

pip install -r requirements.txt

(Note: Please ensure PyTorch is installed correctly with GPU support for your CUDA version before installing other dependencies.)

## Usage

### 1. Data Preparation
Before running the scripts, please update the file paths in the source code to point to your local directories and desired file names:
- In hw2.py: Update train_dir, train_json, val_dir, and val_json variables.
- In predict.py: Update test_img_dir (test images path), model_path (path to your trained weights), and output_json (e.g., pred.json for CodaBench submission).

### 2. Training
To train the model, execute the following command:

python hw2.py

This script will train the model for 80 epochs utilizing CosineAnnealingLR and automatic mixed precision (AMP).

### 3. Inference
To run inference and generate the final predictions on the test set, execute:

python predict.py

The script will load the specified weights, process the test images, and output the bounding box predictions in COCO format to your specified output JSON file.

## Performance Snapshot

<img width="1118" height="281" alt="image" src="https://github.com/user-attachments/assets/dd089607-ff94-4aaa-9c82-ad601bcdb2d8" />

