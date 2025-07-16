import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

cat_dir = os.path.join("TASK 3", "PetImages", "Cat")
dog_dir = os.path.join("TASK 3", "PetImages", "Dog")
print('Cat dir sample:', os.listdir(cat_dir)[:5])
print('Dog dir sample:', os.listdir(dog_dir)[:5])

img_size = (64, 64)
samples_per_class = 100

X = []
y = []

def load_images_from_folder(folder, label, max_samples, X, y):
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
    return X, y

X, y = load_images_from_folder(cat_dir, 0, samples_per_class, X, y)
X, y = load_images_from_folder(dog_dir, 1, samples_per_class, X, y)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0,1],["Cat","Dog"])
plt.yticks([0,1],["Cat","Dog"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=16)
plt.tight_layout()
plt.show()

# Show a grid of 16 test images with predicted and true labels
plt.figure(figsize=(10,10))
for i in range(16):
    idx = i
    img = X_test[idx].reshape(img_size)
    plt.subplot(4,4,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"P:{'Dog' if y_pred[idx] else 'Cat'} / T:{'Dog' if y_test[idx] else 'Cat'}")
    plt.axis('off')
plt.tight_layout()
plt.show() 