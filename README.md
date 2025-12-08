# Cervical Spine Fracture Detection Using Computer Vision
#### This project is a part of the AAI-590 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).
**Status**: Active

## Installation
### Clone the repository:
```bash
git clone https://github.com/apmalinsky/AAI-590-Capstone.git
cd AAI-590-Capstone
```

### ⬇️ Downloading Your Kaggle API Key

To access competition and dataset files programmatically, you must first download your personal **Kaggle API key**.

---

### Steps to Get Your `kaggle.json` File

Follow these steps to generate and download your key:

1.  **Log in** to your Kaggle account on the website.
2.  Navigate to your **Settings** page (Click your profile picture in the top-right corner, then select **Settings**).
3.  Scroll down the page until you find the **Legacy API Credentials** section.
4.  Click the **"Create Legacy API Key"** button.

> 💡 **Result:** This action automatically downloads a file named **`kaggle.json`** to your computer. This file contains your credentials and is essential for authentication.

## Project Intro
This project applies deep learning and computer vision methods to automate the detection of cervical spine fractures in CT scan images. Cervical fractures can be subtle and difficult to diagnose consistently, and missed injuries may result in paralysis, chronic pain, or long-term neurological complications.

The goal is to develop, evaluate, and compare multiple model architectures, including a baseline CNN classifier, Faster R-CNN, YOLO, and DETR, to determine their effectiveness in identifying fracture presence and localizing fractures with bounding boxes. This work contributes towards AI-assisted diagnostic tools that may improve speed, consistency, and clinical workflow efficiency in medical imaging.

## Presentation
Feel free to checkout our [presentation](https://www.youtube.com/watch?v=rmchW63IFFU) walking through the project!

## Contributors
Team 4 — Master of Science in Applied Artificial Intelligence, University of San Diego
* Andy Malinsky 
* Christopher Alleyne
* Jory Hamilton
* Devin Eror

## Methods Used
* Computer Vision
* Deep Learning
* Object Detection
* Transformer-based Vision Models
* Data Preprocessing & Normalization
* Exploratory Data Analysis (EDA)
* Ethics in AI, Healthcare Applications

## Technologies
* Python
* Jupyter Notebook
* Convolutional Neural Networks (CNNs)
* PyTorch
* TorchVision
* Hugging Face Transformers
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn
* GPU acceleration via CUDA (if applicable)
* Kaggle API (dataset acquisition)
* Kaggle Notebook

## Project Description
This project uses a curated subset of the RSNA 2022 Cervical Spine Fracture Detection dataset. The dataset consists of 28,868 CT scan slices, each labeled as fractured or non-fractured, with bounding boxes provided for fractured slices.

### Data Setup
* Navigate to [Kaggle](https://www.kaggle.com/) and click "+ Create" on the left-hand side, then select "Notebook".
* There, you can import [Notebook 00](00_CreatePrivateDataset.ipynb) to get started.
* This notebook will walk you through step-by-step to create your own private Kaggle dataset.

### Data Preprocessing
* Images resized to 256x256 and normalized to [0, 1]
* Undersampling applied to mitigate severe class imbalance (target ratio 1:3 positive to negative)
* Stratified group shuffle split ensures all slices from a patient remain in a single dataset partition.
* Verified image dimensions, class distribution, bounding-box consistency, and normalization during EDA

### Models Implemented
* [**Simple CNN Baseline**](02_SimpleCNN.ipynb): Binary classifier to establish foundational performance
* [**Faster R-CNN**](04_Faster_R_CNN.ipynb): Two-stage detector with ResNet-50 + FPN backbone; designed for small-object medical detection tasks
* [**DETR (Detection Transformer)**](03_DETR.ipynb): Transformer-based detection model fine-tuned with frozen early backbone layers for stability

Each model was trained, validated, and tested on the same curated dataset, with steps including threshold tuning, early stopping, and bounding-box visualization.

### Challenges Addressed
* High class imbalance
* Subtle fracture features requiring high model sensitivity
* Bounding-box formatting alignment for Faster R-CNN
* Overfitting in high-capacity models like DETR
* Need for clinically meaningful detection (localization, not just classification)

## License
MIT License

Copyright (c) 2025 Andy Malinsky

## Acknowledgments
We thank our instructors at the University of San Diego’s Shiley-Marcos School of Engineering for their support and guidance. Additional thanks to Professor Anna Marbut, M.S., for direction throughout this capstone project.
