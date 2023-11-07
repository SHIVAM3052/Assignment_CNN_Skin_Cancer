# Skin Cancer Detection using CNN

## Overview

- This project aims to develop a Convolutional Neural Network (CNN) model for the accurate detection of skin cancer.
- The dataset used in this project contains images of various skin cancer types.
- The goal is to create a model that can assist dermatologists in diagnosing skin cancer at an early stage.

## Data Collection

- The dataset is accessed by mounting Google Drive using Google Colab.
- It contains images of various skin cancer types, with separate subdirectories for each class.

## Data Preprocessing

### Loading Data

- The data is loaded using TensorFlow's `tf.keras.utils.image_dataset_from_directory` function.
- The dataset is split into training and validation datasets, with an 80% - 20% ratio.

### Data Augmentation

- Data augmentation is applied to address class imbalance and improve model generalization.
- The `Augmentor` library is used to add more samples to classes with fewer images.
- Augmentation techniques include random rotation, flipping, and zooming.

## Model Creation

- A CNN model is created to classify skin cancer types.
- The model architecture consists of convolutional layers, max-pooling layers, and fully connected layers.
- Batch normalization is applied to some layers to improve training stability.

## Model Training

- The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.
- It is trained for 50 epochs to ensure convergence.
- The training process is visualized with accuracy and loss plots.

## Model Evaluation

- The model's performance is evaluated based on training and validation accuracy and loss.
- Results are visualized to assess the model's training progress and identify potential issues.

## Findings

- Initial training of the model shows signs of underfitting.
- Class imbalance is identified as an issue, which prompted the use of data augmentation to balance the classes.
- Data augmentation techniques significantly improve the model's performance and reduce underfitting.
- The final model demonstrates improved accuracy and reduced loss, indicating successful training.

## Conclusion

- This project highlights the importance of data preprocessing, data augmentation, and model training to address class imbalance and improve skin cancer detection using a CNN.
- The code and findings can serve as a foundation for further research and development in medical image analysis.

## Instructions

- The complete code and dataset can be found in the Jupyter Notebook provided in the repository.
- Feel free to experiment with different augmentation techniques, model architectures, and hyperparameters to further enhance the model's performance.
