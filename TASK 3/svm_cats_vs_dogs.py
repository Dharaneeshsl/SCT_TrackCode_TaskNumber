# Install dependencies as needed:
# pip install scikit-learn pillow numpy
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set local dataset directories
cat_dir = os.path.join("TASK 3", "PetImages", "Cat")
dog_dir = os.path.join("TASK 3", "PetImages", "Dog")
print('Cat dir sample:', os.listdir(cat_dir)[:5])
print('Dog dir sample:', os.listdir(dog_dir)[:5])

# Parameters
img_size = (64, 64)
samples_per_class = 100

X = []
y = []

def load_images_from_folder(folder, label, max_samples):
    count = 0
    for filename in os.listdir(folder):
        if count >= max_samples:
            break
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert('L').resize(img_size)
            X.append(np.array(img).flatten())
            y.append(label)
            count += 1
        except Exception:
            continue

# Load 100 cats and 100 dogs
load_images_from_folder(cat_dir, 0, samples_per_class)
load_images_from_folder(dog_dir, 1, samples_per_class)

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
clf = SVC(kernel='linear', max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2f}") 