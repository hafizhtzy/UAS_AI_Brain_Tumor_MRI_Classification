# UAS Artificial Intelligence - Brain Tumor MRI Classification

## Title

**Brain Tumor MRI Classification Using Transfer Learning: A Comparative Study of ResNet50, EfficientNetB0, and MobileNetV2**

## Author

**Muhammad Abdurrahman Hafizhuddin**
Informatics Study Program
Faculty of Science
Universitas Islam Negeri Sultan Maulana Hasanuddin Banten
Serang, Banten, Indonesia

## Project Description

This repository contains the source code, experiment results, and visualization files for the Artificial Intelligence final project. The study focuses on brain tumor MRI image classification using transfer learning models.

The experiment compares three convolutional neural network architectures:

1. ResNet50
2. EfficientNetB0
3. MobileNetV2

The models are trained and evaluated on a brain tumor MRI dataset containing four classes:

* Glioma
* Meningioma
* No-tumor
* Pituitary

## Research Objective

The objective of this project is to replicate and compare transfer learning-based deep learning methods for classifying brain tumor MRI images. The comparison is performed using accuracy, precision, recall, F1-score, confusion matrix, and training history visualization.

## Dataset

The dataset used in this project is a public Brain Tumor MRI dataset. The dataset is not uploaded directly to this GitHub repository because of file size considerations.

The complete dataset is available in the Google Drive submission folder. Link Google Drive dataset [https://drive.google.com/drive/folders/1whV7ic382JWz6A-n_JJ08dxZO09wJENq?usp=sharing]

Dataset structure:

```text
dataset/
├── Training/
│   ├── Glioma/
│   ├── Meningioma/
│   ├── No-tumor/
│   └── Pituitary/
└── Testing/
    ├── Glioma/
    ├── Meningioma/
    ├── No-tumor/
    └── Pituitary/
```

## Repository Structure

```text
UAS_AI_Brain_Tumor_MRI_Classification/
├── README.md
├── code/
│   ├── train_all_local_pytorch.py
│   └── evaluation scripts or supporting files
├── results/
│   ├── classification_report_resnet50.txt
│   ├── classification_report_efficientnetb0.txt
│   ├── classification_report_mobilenetv2.txt
│   ├── comparison_result_final.csv
│   └── comparison_result_final.xlsx
└── visualizations/
    ├── accuracy_curve_resnet50.png
    ├── accuracy_curve_efficientnetb0.png
    ├── accuracy_curve_mobilenetv2.png
    ├── loss_curve_resnet50.png
    ├── loss_curve_efficientnetb0.png
    ├── loss_curve_mobilenetv2.png
    ├── confusion_matrix_resnet50.png
    ├── confusion_matrix_efficientnetb0.png
    ├── confusion_matrix_mobilenetv2.png
    └── comparison graphs
```

## Methodology

The research workflow consists of the following stages:

1. Dataset collection and organization
2. Image preprocessing
3. Data splitting into training, validation, and testing sets
4. Model training using transfer learning
5. Model evaluation on the testing dataset
6. Comparison of ResNet50, EfficientNetB0, and MobileNetV2
7. Visualization of training curves and confusion matrices
8. Preparation of IEEE Access-style article draft

## Preprocessing

The preprocessing steps include:

* Loading MRI images from the dataset folders
* Converting images into RGB format
* Resizing images to 224 × 224 pixels
* Converting images into tensors
* Normalizing image tensors using ImageNet normalization values
* Applying light augmentation to training data

The preprocessing process is performed during runtime using PyTorch transforms and DataLoader.

## Models Used

### ResNet50

ResNet50 is used as a transfer learning model with pretrained ImageNet weights. The final classification layer is modified to classify four MRI classes.

### EfficientNetB0

EfficientNetB0 is used as a lightweight and efficient transfer learning model. It provides a balance between accuracy and computational efficiency.

### MobileNetV2

MobileNetV2 is used as a lightweight model suitable for faster inference and lower computational cost.

## Experimental Results

The final evaluation results show that the transfer learning models achieved high classification performance on the testing dataset.

| Model          | Test Accuracy | Precision | Recall | F1-Score |
| -------------- | ------------: | --------: | -----: | -------: |
| ResNet50       |        0.9939 |    0.9939 | 0.9939 |   0.9939 |
| EfficientNetB0 |        0.9954 |    0.9954 | 0.9954 |   0.9954 |
| MobileNetV2    |        0.9916 |    0.9917 | 0.9916 |   0.9916 |

Based on the results, EfficientNetB0 achieved the highest test accuracy, while ResNet50 and MobileNetV2 also produced strong classification performance.

## Output Files

The project produces several output files:

* Trained model files
* Classification reports
* Comparison result table
* Accuracy curves
* Loss curves
* Confusion matrices
* Final IEEE Access article draft
* Presentation file
* Turnitin report

Large files such as datasets and trained model checkpoints are stored in the Google Drive submission folder.

## Environment

The experiment was conducted using:

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* NVIDIA CUDA GPU

Local device used for training:

* Intel Core i9-14900HX
* NVIDIA GeForce RTX 4070 Laptop GPU 8GB
* 32GB RAM

## How to Run

Install the required libraries:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow openpyxl
```

Run the training script:

```bash
python code/train_all_local_pytorch.py
```

After training, the results will be saved into:

```text
results/
visualizations/
models/
```

## Notes

The dataset and trained model files are not included in this GitHub repository because of file size limitations. They are available in the Google Drive submission folder.

This repository is part of an academic final project for the Artificial Intelligence course.

## License

This repository is created for academic and educational purposes only.
