import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

def load_hagrid_dataset(data_dir):
    """Load and preprocess HaGrid dataset"""
    print("Loading HaGrid dataset...")
    
    # HaGrid gesture classes
    gesture_classes = [
        'call', 'dislike', 'fist', 'four', 'like', 'mute', 
        'ok', 'one', 'palm', 'peace', 'rock', 'stop', 
        'stop_inverted', 'three', 'two_up', 'two_up_inverted'
    ]
    
    X, y = [], []
    
    # Look for images in hagrid dataset structure
    dataset_path = os.path.join(data_dir, 'hagrid')
    
    if os.path.exists(dataset_path):
        # Try to find images in the dataset
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract gesture label from path or filename
                    file_path = os.path.join(root, file)
                    
                    # Try to determine gesture from path
                    path_parts = file_path.lower().split(os.sep)
                    gesture = None
                    
                    for part in path_parts:
                        if any(cls in part for cls in gesture_classes):
                            for cls in gesture_classes:
                                if cls in part:
                                    gesture = cls
                                    break
                            break
                    
                    if gesture:
                        try:
                            # Load and preprocess image
                            img = cv2.imread(file_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                                img = img / 255.0  # Normalize
                                
                                X.append(img)
                                y.append(gesture)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
    
    # If no HaGrid data found, create synthetic data for demonstration
    if len(X) == 0:
        print("No HaGrid data found, creating synthetic training data...")
        return create_synthetic_data()
    
    print(f"Loaded {len(X)} images from HaGrid dataset")
    return np.array(X), np.array(y)

def create_synthetic_data():
    """Create synthetic data for demonstration if no dataset is available"""
    print("Creating synthetic gesture data for training...")
    
    # Create synthetic data with different patterns for each gesture
    gestures = ['fist', 'palm', 'peace', 'ok', 'one', 'two_up', 'three', 'four']
    X, y = [], []
    
    for gesture in gestures:
        for i in range(100):  # 100 samples per gesture
            # Create synthetic image with different patterns
            img = np.random.rand(IMG_SIZE, IMG_SIZE, 3)
            
            # Add gesture-specific patterns
            if gesture == 'fist':
                img[:, :, 0] *= 0.3  # Darker red channel
            elif gesture == 'palm':
                img[:, :, 1] *= 1.2  # Brighter green channel
            elif gesture == 'peace':
                img[50:150, 50:150, 2] = 0.8  # Blue square pattern
            elif gesture == 'ok':
                cv2.circle(img, (IMG_SIZE//2, IMG_SIZE//2), 30, (1, 1, 1), -1)
            elif gesture == 'one':
                img[IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//2-10:IMG_SIZE//2+10, :] = 0.9
            elif gesture == 'two_up':
                img[IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//2-20:IMG_SIZE//2, :] = 0.9
                img[IMG_SIZE//4:3*IMG_SIZE//4, IMG_SIZE//2+10:IMG_SIZE//2+30, :] = 0.9
            elif gesture == 'three':
                for j in range(3):
                    x_pos = IMG_SIZE//2 + (j-1)*20
                    img[IMG_SIZE//4:3*IMG_SIZE//4, x_pos-5:x_pos+5, :] = 0.9
            elif gesture == 'four':
                for j in range(4):
                    x_pos = IMG_SIZE//2 + (j-1.5)*15
                    img[IMG_SIZE//4:3*IMG_SIZE//4, x_pos-3:x_pos+3, :] = 0.9
            
            X.append(img)
            y.append(gesture)
    
    return np.array(X), np.array(y)

def create_cnn_model(num_classes):
    """Create a CNN model for gesture recognition"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the gesture recognition model"""
    print("Starting gesture recognition model training...")
    
    # Load dataset
    data_dir = r"C:\Users\BALA NITHYA SREE S\Desktop\SkillCraft\TASK 4"
    X, y = load_hagrid_dataset(data_dir)
    
    if len(X) == 0:
        print("No data available for training!")
        return
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    print(f"Dataset: {len(X)} samples, {num_classes} classes")
    print(f"Classes: {list(le.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create and train model
    model = create_cnn_model(num_classes)
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Save model and label encoder
    model.save('gesture_cnn_model.h5')
    joblib.dump(le, 'gesture_label_encoder.joblib')
    
    # Save class names for easy access
    with open('gesture_classes.json', 'w') as f:
        json.dump(list(le.classes_), f)
    
    print("Model saved successfully!")
    print("Files created:")
    print("- gesture_cnn_model.h5 (trained model)")
    print("- gesture_label_encoder.joblib (label encoder)")
    print("- gesture_classes.json (class names)")
    
    return model, le, history

if __name__ == "__main__":
    train_model()
