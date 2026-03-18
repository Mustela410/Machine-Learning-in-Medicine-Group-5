# Breast Ultrasound Segmentation and Classification

## Project Overview
This project uses Deep Learning to analyze breast ultrasound images. It focuses on two main tasks:
1. **Segmentation**: Finding the exact location and boundary of a tumor.
2. **Classification**: Diagnosing the tumor as Normal, Benign, or Malignant.

To solve this, we apply two different approaches. The first approach handles each task separately using independent models. The second approach uses a single, unified model to perform both tasks at the same time.

## Code Structure
The code is organized into three main Jupyter Notebooks along with their python file version:

* **`Segmentation.ipynb`**: A standalone U-Net model. Its only job is to find the tumor and draw a mask around it.
* **`Classification.ipynb`**: A standalone ResNet34 model. It analyzes the images and categorizes them into their clinical types.
* **`Multi-task.ipynb`**: A combined model that handles both segmentation and classification together. This notebook also uses Optuna to automatically balance the training process between the two tasks.

## Dataset 
We trained and tested the models using the Breast Ultrasound Images dataset. The data is grouped into three categories:
* Normal (No tumor)
* Benign (Non-cancerous tumor)
* Malignant (Cancerous tumor)