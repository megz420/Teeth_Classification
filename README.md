# Teeth Classification

This project trains a TensorFlow image classifier on the included teeth image
dataset.

## Dataset

The dataset is in `Teeth_Dataset/`, organized into `Training`, `Validation`,
and `Testing` folders. Each split contains subfolders for the image classes.

## Run the notebook

Install TensorFlow, Matplotlib, and NumPy, then open `TeethMain.ipynb` from the
repository root. The notebook reads training and validation images from the
relative paths `Teeth_Dataset/Training` and `Teeth_Dataset/Validation`.
