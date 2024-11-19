# Digit Recognition using CNN

This project involves building a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is trained and saved, and then used to predict digits drawn by the user in MS Paint.

## Table of Contents

- [Project Description](#project-description)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Project Description

The goal of this project is to create a CNN model that can recognize handwritten digits. The model is trained on the MNIST dataset and then used to predict digits drawn by the user in MS Paint. The user can draw a digit, save the image, and the model will predict the digit.

## Setup and Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MuhammadMehdiRaza/Hand_Written_Digit_Recogntition.git
   cd Hand_Written_Digit_Recogntition
   ```

2. **Ensure Python and Related Library Installation**
   ```bash
   pip install tensorflow matplotlib cv2 python numpy subprocess os
   ```

## Usage

1. **Open MS Paint**

   The script will automatically open MS Paint. Draw a digit and save the image in the **Digits_Folder** directory with the name format **{image_no}.png.**

2. **Predict the Digit (0-9)**

   After saving the image, the script will predict the digit and display the result along with the image.

3. **Continue or Exit**

   The script will ask if you want to continue. If you choose to continue, the process will repeat. If you choose to exit, the script will terminate.

## Model Architecture

The CNN model consists of the following layers:

- Conv2D layer with 64 filters, kernel size 3, and ReLU activation.

- Conv2D layer with 32 filters, kernel size 3, and ReLU activation.

- MaxPooling2D layer with pool size (2, 2).
- Flatten layer.

- Dense layer with 10 units and softmax activation

The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy metric.

## Known Limitations

- The dataset comprises from 0-9.

- The script assumes the user saves the image in the correct format and location: **Digits_Folder/{image_no}.png.**

- Input images must be **GrayScale** and **28x28 pixels** have to be selected when **Paint** gets opened for optimal performance.

## Future Enhancements

- Replace MS Paint with an integrated GUI for drawing.

- Add preprocessing steps to handle varying input image sizes and formats.

- Extend the system for multiclass classification of custom datasets.

## Contributing

Contributions to enhance the project are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- **Keras and TensorFlow:** For providing an easy-to-use framework for deep learning.

- **MNIST Dataset:** For serving as a reliable benchmark for digit recognition tasks.

- **OpenCV:** For image processing functionality.

- **Chat GPT and Youtube:** Now a days, with these tools, learning rate has progressively improved compared to past. I encourage everyone to take benefit.

- **Machine Learning Specialization Course:** Do check out this course from Courseera. Best for beginners to start with Machine Learning.
