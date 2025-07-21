import cv2
import numpy as np
import tensorflow as tf
import joblib
import json
import mediapipe as mp

# Load trained model and encoders
try:
    model = tf.keras.models.load_model('gesture_cnn_model.h5')
    le = joblib.load('gesture_label_encoder.joblib')
    with open('gesture_classes.json', 'r') as f:
        class_names = json.load(f)
    print("Loaded trained CNN model successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run improved_gesture_model.py first to train the model.")
    exit()

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

IMG_SIZE = 224

def preprocess_hand_region(image, landmarks):
    """Extract and preprocess hand region from image"""
    h, w, _ = image.shape
    
    # Get bounding box of hand
    x_coords = [landmark.x for landmark in landmarks.landmark]
    y_coords = [landmark.y for landmark in landmarks.landmark]
    
    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
    
    # Add padding
    padding = 30
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Extract hand region
    hand_region = image[y_min:y_max, x_min:x_max]
    
    if hand_region.size == 0:
        return None
    
    # Resize and normalize
    hand_region = cv2.resize(hand_region, (IMG_SIZE, IMG_SIZE))
    hand_region = hand_region / 255.0
    
    return hand_region, (x_min, y_min, x_max, y_max)

def predict_gesture(hand_region):
    """Predict gesture using trained CNN model"""
    if hand_region is None:
        return "No Hand", 0.0
    
    # Prepare input for model
    input_data = np.expand_dims(hand_region, axis=0)
    
    # Make prediction
    predictions = model.predict(input_data, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Get gesture name
    gesture_name = le.inverse_transform([predicted_class])[0]
    
    return gesture_name, confidence

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("CNN-based Hand Gesture Recognition")
print("Trained model loaded successfully!")
print("Press 'q' to quit")
print(f"Recognizable gestures: {', '.join(class_names)}")

# For smoothing predictions
prediction_history = []
history_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
    
    gesture_name = "No Hand"
    confidence = 0.0
    bbox = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Extract and predict gesture
            hand_data = preprocess_hand_region(rgb_frame, hand_landmarks)
            if hand_data is not None:
                hand_region, bbox = hand_data
                gesture_name, confidence = predict_gesture(hand_region)
                
                # Add to prediction history for smoothing
                prediction_history.append(gesture_name)
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
                
                # Use most common prediction in recent history
                if len(prediction_history) >= 3:
                    from collections import Counter
                    most_common = Counter(prediction_history).most_common(1)[0][0]
                    gesture_name = most_common
    
    # Draw bounding box around hand region
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    
    # Display prediction with confidence
    if confidence > 0.5:  # Only show if confident
        text = f'{gesture_name.upper()} ({confidence:.2f})'
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)  # Green if very confident, orange otherwise
    else:
        text = "Uncertain"
        color = (0, 0, 255)  # Red for uncertain
    
    # Draw text with background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Background rectangle
    cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 35), font, font_scale, color, thickness)
    
    # Show additional info
    info_text = f"Model: CNN | Classes: {len(class_names)}"
    cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display frame
    cv2.imshow('CNN Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
