"""
Test script to verify the setup and model
Run this after training to ensure everything works correctly
"""

import os
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'streamlit': 'Streamlit',
        'streamlit_drawable_canvas': 'Streamlit Drawable Canvas',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!\n")
    return True

def check_model():
    """Check if the trained model exists"""
    print("Checking for trained model...")
    
    model_path = "models/digit_recognition_model.keras"
    
    if not os.path.exists("models"):
        print("❌ 'models' directory not found")
        os.makedirs("models")
        print("✅ Created 'models' directory")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("⚠️  Please run: python train_model.py")
        return False
    
    print(f"✅ Model found at: {model_path}")
    
    # Try to load the model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_prediction():
    """Test a sample prediction"""
    print("\nTesting prediction with sample data...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        # Load model
        model = tf.keras.models.load_model("models/digit_recognition_model.keras")
        
        # Create a sample image (zeros)
        sample_img = np.zeros((1, 28, 28, 1))
        
        # Make prediction
        prediction = model.predict(sample_img, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100
        
        print(f"✅ Sample prediction successful")
        print(f"   Predicted digit: {predicted_digit}")
        print(f"   Confidence: {confidence:.2f}%")
        return True
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    
    dirs = ['models', 'saved_digits']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("  Hand Written Digit Recognition - Setup Verification")
    print("=" * 60)
    print()
    
    results = {
        "Dependencies": check_dependencies(),
        "Directories": check_directories(),
        "Model": check_model(),
    }
    
    if all(results.values()):
        results["Prediction"] = test_prediction()
    
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test:.<40} {status}")
    
    print("=" * 60)
    
    if all(results.values()):
        print("\n🎉 All tests passed! You're ready to run the app!")
        print("   Run: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        if not results["Model"]:
            print("   Run: python train_model.py")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
