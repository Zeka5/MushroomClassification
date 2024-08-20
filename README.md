# Mushroom Classification Using CNN, SVM, and Transfer Learning

## Problem Description

This project involves classifying images of mushrooms into one of nine categories using two different algorithms. The categories represent some of the most common genera found in Northern Europe.

## Motivation

Mushroom foraging is a popular hobby, but it can be challenging for those without expertise. Accurately identifying mushrooms from images is also difficult due to the large number of species, their similar appearances, and varying environmental conditions in photos. The goal of this project is to create a program that automates mushroom classification, making it easier for people to identify them correctly.

## Dataset

The dataset used in this project is available on Kaggle:

- [Mushrooms Classification - Common Genus Images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)

This dataset includes nine folders, each containing between 300 and 1600 images of mushrooms from different genera. The categories are as follows:

- **Agaricus**: 353 images
- **Amanita**: 750 images
- **Boletus**: 1073 images
- **Cortinarius**: 836 images
- **Entoloma**: 364 images
- **Hygrocybe**: 316 images
- **Lactarius**: 1563 images
- **Russula**: 1148 images
- **Suillus**: 311 images

## Data Preprocessing

Before processing, the images will undergo data augmentation by performing horizontal flips. This will increase the dataset size and improve model accuracy.

## Methodology

To solve this problem, we will use:

1. **Convolutional Neural Networks (CNN)**: A deep learning algorithm commonly used for image classification.
2. **Support Vector Machines (SVM)**: A machine learning algorithm suitable for classification tasks.
3. **Transfer Learning**: Using a pre-trained model and fine-tuning it for mushroom classification.

The steps involved are:

- Data preparation
- Model training
- Model evaluation

## Evaluation

The models will be evaluated based on classification accuracy, i.e., the percentage of correctly classified images. The dataset will be split as follows:

- **Training Data**: 80%
- **Validation Data**: 10%
- **Test Data**: 10%

We will compare the performance of different models (CNN, SVM, and the transfer learning model) and analyze their accuracy. The analysis will include identifying misclassified classes and comparing the performance of the different algorithms.

## Technologies

The following technologies and programming languages will be used:

- **Python**
- Libraries for CNN, SVM, and ResNet for transfer learning
