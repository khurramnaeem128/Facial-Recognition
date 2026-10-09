# 👤 Facial Recognition Using Machine Learning

## 📌 Project Overview

The Facial Recognition project applies computer vision and machine learning techniques to recognize faces from images. It uses Histogram of Oriented Gradients (HOG) for facial feature extraction and a Support Vector Machine (SVM) classifier to identify individuals based on extracted image features.

The project demonstrates a practical machine learning workflow involving image data preparation, feature extraction, model training, classification, and performance evaluation.

The implementation uses a subset of the Labeled Faces in the Wild (LFW) dataset to explore how machine learning can be applied to facial recognition.

## 🎯 Project Objectives

* Work with facial image data.
* Prepare images for machine learning.
* Extract facial image features using HOG.
* Train a Support Vector Machine classifier.
* Recognize individuals based on extracted features.
* Evaluate model performance using classification accuracy.
* Understand the challenges and limitations of facial recognition systems.

## 📊 Dataset

The project uses a subset of the **Labeled Faces in the Wild (LFW)** dataset, a collection of face photographs commonly used for face recognition research.

The project worked with a subset of approximately 1,300 images, with the actual sample count depending on the dataset loading and filtering configuration.

Dataset preparation includes:

* Loading facial images and their identity labels.
* Preparing images for feature extraction.
* Extracting numerical representations of facial appearance.
* Preparing features and labels for classifier training and evaluation.

The experiment used a subset of the available dataset rather than the complete LFW collection.

## 🛠️ Technologies Used

| Technology                   | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| Python                       | Programming and implementation                   |
| NumPy                        | Numerical operations                             |
| Scikit-learn                 | Dataset handling, model training, and evaluation |
| HOG                          | Facial image feature extraction                  |
| Support Vector Machine (SVM) | Face classification                              |
| LFW Dataset                  | Facial image data                                |

## ⚙️ Computer Vision Workflow

### 1. Dataset Loading

Loaded a subset of facial images and corresponding identity labels from the LFW dataset.

### 2. Image Preprocessing

Prepared the images for feature extraction and machine learning.

### 3. Feature Extraction

Applied Histogram of Oriented Gradients (HOG) to convert facial images into numerical feature vectors.

HOG captures information about local image gradients and edge orientations, providing a representation of facial appearance.

### 4. Feature Preparation

Prepared the extracted feature vectors and their corresponding identity labels for classification.

### 5. Model Training

Trained a Support Vector Machine classifier using the extracted facial features.

### 6. Face Classification

Used the trained classifier to predict the identity class associated with an input feature vector.

### 7. Model Evaluation

Evaluated recognition performance using classification accuracy on the prepared evaluation dataset.

## 🤖 Model Used: Support Vector Machine (SVM)

Support Vector Machine is a supervised machine learning algorithm that learns decision boundaries to distinguish between classes.

In this project, a linear SVM classifier was used to classify facial images based on their extracted HOG features.

### Model Configuration

| Parameter                      | Value                                  |
| ------------------------------ | -------------------------------------- |
| Dataset                        | Labeled Faces in the Wild (LFW) subset |
| Feature Extraction             | Histogram of Oriented Gradients (HOG)  |
| Feature Vector Size            | 900                                    |
| Classifier                     | Linear SVM                             |
| Regularization Parameter (`C`) | 0.1                                    |
| Recognition Threshold          | 12                                     |

The recognition threshold was part of the reported project configuration. Its exact interpretation depends on how it was implemented in the code.

## 📈 Model Performance

The project achieved the following reported result:

| Evaluation Metric    | Result |
| -------------------- | -----: |
| Recognition Accuracy | 68.59% |

### Performance Analysis

**Recognition Accuracy — 68.59%**

The classifier correctly predicted approximately 68.59% of the samples in the evaluated dataset.

This result provides a baseline for the implemented HOG and linear SVM approach. Performance can vary depending on the selected identities, number of training examples per person, image quality, and train-test splitting strategy.

Accuracy alone does not establish that a facial recognition system is reliable for real-world identity verification.

## 💡 Key Learnings

Through this project, I gained practical experience with:

* Working with facial image datasets.
* Understanding computer vision workflows.
* Extracting image features using HOG.
* Converting images into numerical feature vectors.
* Training a linear SVM classifier.
* Applying supervised machine learning to face recognition.
* Evaluating classification accuracy.
* Understanding the limitations of image-based recognition.

## 🔍 Potential Applications

Facial recognition techniques can be explored in areas such as:

* Computer vision research.
* Image classification experiments.
* Face image organization.
* Educational demonstrations of identity classification.
* Research into visual feature extraction and recognition.

Real-world applications require careful evaluation of privacy, consent, security, fairness, and recognition errors.

## 🚀 Future Improvements

* Experiment with different HOG configurations.
* Tune SVM hyperparameters.
* Evaluate performance across different train-test splits.
* Analyze precision, recall, and confusion matrices.
* Compare HOG features with alternative feature extraction methods.
* Explore dimensionality reduction where appropriate.
* Test robustness under different lighting, pose, and image-quality conditions.
* Investigate modern deep learning-based face representations.
* Evaluate performance across different demographic groups and operating conditions.

## 📂 Project Structure

```text
Facial-Recognition/
│
├── facial_recognition.py
└── README.md
```

Adjust the filenames and folder structure to match the actual project files. Include the dataset only if redistribution is permitted.

## ▶️ Getting Started

### Prerequisites

Python must be installed on your system.

### 1. Install Dependencies

```bash
python -m pip install numpy scikit-learn scikit-image
```

### 2. Run the Project

```bash
python facial_recognition.py
```

Replace `facial_recognition.py` with the actual Python script name if it differs.

If the script downloads or loads the LFW dataset automatically, ensure that the required dataset is available in your environment.

## 🔐 Privacy and Responsible Use

Facial images contain sensitive biometric information. This project is intended for educational and experimental purposes.

Any real-world facial recognition application should consider informed consent, secure handling of biometric data, fairness, false matches, and applicable privacy requirements.

This implementation has not been established as a production-ready identity verification system.

## ✅ Conclusion

This project demonstrates how computer vision and supervised machine learning can be combined to perform facial recognition. HOG feature extraction and a linear SVM classifier were used to process facial images and classify identities, achieving a reported accuracy of approximately 68.59%.

The project provides a foundation for further experimentation with feature extraction, classifier optimization, and more advanced face recognition techniques.
