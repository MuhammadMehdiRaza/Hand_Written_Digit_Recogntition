# 🎨 Handwritten Digit Recognition with CNN & Streamlit

An interactive web application for recognizing handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset. Draw digits directly in your browser and get real-time predictions!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)

## ✨ Features

- 🎨 **Interactive Drawing Canvas**: Draw digits directly in your browser
- 🤖 **Real-time Predictions**: Instant digit recognition with confidence scores
- 📊 **Probability Distribution**: See prediction probabilities for all digits (0-9)
- 💾 **Save Drawings**: Save your drawings for later use
- 🎯 **High Accuracy**: CNN model trained on 60,000 MNIST samples
- 🎨 **Customizable UI**: Adjust brush size, colors, and more

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MuhammadMehdiRaza/Hand_Written_Digit_Recogntition.git
   cd Hand_Written_Digit_Recogntition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time only)
   ```bash
   python train_model.py
   ```
   This will train the CNN model and save it to the `models/` directory. Training takes about 5-10 minutes depending on your hardware.

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** at `http://localhost:8501` and start drawing!

## 📁 Project Structure

```
Hand_Written_Digit_Recogntition/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── 3.py                        # Original MS Paint-based version
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── models/                     # Saved models directory
│   ├── digit_recognition_model.keras
│   └── training_history.png
└── saved_digits/              # Saved user drawings
```

## 🧠 Model Architecture

The CNN model consists of:

- **Conv2D Layer 1**: 64 filters, 3×3 kernel, ReLU activation
- **Conv2D Layer 2**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Dropout**: 25% dropout rate
- **Flatten Layer**: Converts 2D features to 1D
- **Dense Layer 1**: 128 neurons, ReLU activation
- **Dropout**: 50% dropout rate
- **Dense Layer 2**: 10 neurons (output), Softmax activation

**Performance:**
- Training Accuracy: ~99%
- Test Accuracy: ~98%
- Dataset: MNIST (60,000 training, 10,000 test images)

## 🎯 Usage Guide

### Drawing Tips for Best Results

1. **Center your digit**: Draw in the middle of the canvas
2. **Use thick strokes**: Increase brush size for clearer digits
3. **Make it bold**: The model works best with clear, bold digits
4. **Fill the space**: Don't make your digit too small
5. **Try different styles**: The model is trained on various handwriting styles

### Streamlit App Features

- **Canvas Settings**: Adjust brush size and colors in the sidebar
- **Clear Canvas**: Reset the canvas to draw a new digit
- **Save Drawing**: Save your drawings to the `saved_digits/` folder
- **Live Preview**: See the 28×28 preprocessed image
- **Probability Chart**: View confidence scores for all digits

## 🔧 Advanced Usage

### Retrain the Model

To retrain with different parameters, edit `train_model.py`:

```python
# Adjust epochs
hist = model.fit(x_train, y_train_one_hot, epochs=20)  # Increase epochs

# Modify architecture
model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
```

### Use Your Own Images

You can test the model with your own images:

```python
import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/digit_recognition_model.keras")
img = cv2.imread("your_image.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = cv2.bitwise_not(img)  # Invert if needed
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

prediction = model.predict(img)
print(f"Predicted digit: {np.argmax(prediction)}")
```

## 📦 Dependencies

- **TensorFlow**: Deep learning framework
- **Streamlit**: Web app framework
- **streamlit-drawable-canvas**: Canvas widget for Streamlit
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Pillow**: Image handling
- **Matplotlib**: Plotting and visualization

## 🐛 Troubleshooting

### Model not found error
```bash
# Make sure to train the model first
python train_model.py
```

### Canvas not displaying
```bash
# Reinstall streamlit-drawable-canvas
pip install --upgrade streamlit-drawable-canvas
```

### TensorFlow issues
```bash
# For Apple Silicon (M1/M2)
pip install tensorflow-macos

# For GPU support
pip install tensorflow-gpu
```

## 🎓 How It Works

1. **Data Loading**: Loads 70,000 MNIST images (28×28 grayscale)
2. **Preprocessing**: Normalizes pixel values and one-hot encodes labels
3. **Training**: Trains CNN for 10 epochs with Adam optimizer
4. **Prediction**: Converts canvas drawing to 28×28, preprocesses, and predicts
5. **Visualization**: Displays prediction with confidence scores

## 🚀 Future Enhancements

- [ ] Support for multi-digit recognition
- [ ] Model comparison (different architectures)
- [ ] Export predictions to CSV
- [ ] Mobile-responsive design
- [ ] Dark/Light theme toggle
- [ ] Batch prediction from uploaded images
- [ ] Real-time training visualization
- [ ] Support for custom datasets

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, and Christopher Burges
- **TensorFlow/Keras**: Google Brain Team
- **Streamlit**: Streamlit Team
- Original project by [Muhammad Mehdi Raza](https://github.com/MuhammadMehdiRaza)

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

Made with ❤️ using Python, TensorFlow, and Streamlit

- Extend the system for multiclass classification of custom datasets.

## Contributing

Contributions to enhance the project are welcome! Please fork the repository and submit a pull request with your changes.

## Acknowledgments

- **Keras and TensorFlow:** For providing an easy-to-use framework for deep learning.

- **MNIST Dataset:** For serving as a reliable benchmark for digit recognition tasks.

- **OpenCV:** For image processing functionality.

- **Chat GPT and Youtube:** Now a days, with these tools, learning rate has progressively improved compared to past. I encourage everyone to take benefit.

- **Machine Learning Specialization Course:** Do check out this course from Courseera. Best for beginners to start with Machine Learning.
