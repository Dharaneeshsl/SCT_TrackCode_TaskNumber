import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

data_dir = r"C:\Users\BALA NITHYA SREE S\Desktop\SkillCraft\TASK 4\leapGestRecog"
img_size = (64, 64)

X, y = [], []

for subject in os.listdir(data_dir):
    subject_path = os.path.join(data_dir, subject)
    if not os.path.isdir(subject_path):
        continue
    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)
        if not os.path.isdir(gesture_path):
            continue
        label = gesture
        for img_file in os.listdir(gesture_path):
            if img_file.endswith('.png'):
                img_path = os.path.join(gesture_path, img_file)
                img = Image.open(img_path).convert('L').resize(img_size)
                X.append(np.array(img).flatten())
                y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.preprocessing import LabelEncoder
import joblib

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

clf = SVC(kernel='linear', max_iter=1000, probability=True)
clf.fit(X_train, y_train_enc)

# Save model and label encoder
joblib.dump(clf, 'svm_hand_gesture_model.joblib')
joblib.dump(le, 'label_encoder_hand_gesture.joblib')

y_pred = clf.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_labels))

plt.figure(figsize=(8,6))
plt.imshow(confusion_matrix(y_test, y_pred), cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show() 