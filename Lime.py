#Importing Libraries
import os
import numpy as np
import cv2
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from tensorflow.keras import layers, optimizers, callbacks, Model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import mixed_precision#Setting Configurations

#Setting Configurations
mixed_precision.set_global_policy('mixed_float16')
#all the details to help training the model 
IMG_SIZE = 224
NUM_CLASSES = 2
MAX_PER_CLASS = 10700
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 2e-4
KFOLDS = 5
DATA_PATH = "/mnt/home/belhraka/Ali_venv/COVID-19_Radiography_Dataset"
RESULTS_DIR = "./results_effnetb7"
os.makedirs(RESULTS_DIR, exist_ok=True)

#Loading Dataset
file_paths, labels = [], []
#sorting image one by one
for cls_name in sorted(os.listdir(DATA_PATH)):
    cls_dir = os.path.join(DATA_PATH, cls_name)
    imgs = os.listdir(cls_dir)
    #when it sort the images it shuffle it , this is helping for refreshing the model 
    random.shuffle(imgs)
    for fname in imgs[:MAX_PER_CLASS]:
        #for each name or let's say a class in each image 
        file_paths.append(os.path.join(cls_dir, fname))
        labels.append(cls_name)

#encoding the classes 
le = LabelEncoder()
y_all = le.fit_transform(labels)
#making the images into arrays
paths = np.array(file_paths)
#making the classes into arrays
y_all = np.array(y_all)
#resize the images
images = np.array([cv2.resize(cv2.imread(p), (IMG_SIZE, IMG_SIZE)) for p in paths], dtype='float32') / 255.0


#splitting the data
X_temp, X_test, y_temp, y_test = train_test_split(images, y_all, test_size=0.10, stratify=y_all, random_state=42)
X, y = X_temp, y_temp


#Loading The Model 
from tensorflow.keras.models import load_model

MODEL_PATH = "/mnt/projects/sutravek_project/Ali_belhrak/Next_Step/final_model.keras"
final_model = load_model(MODEL_PATH)

#LIME METHOD
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==== 1. Preprocess image ====
img_path = "/mnt/home/belhraka/Ali_venv/COVID-19_Radiography_Dataset/COVID/COVID-10.png"
IMG_SIZE = 224  # adjust to your model input

# Read grayscale CT and resize
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

# Equalize contrast if needed (like your img_eq variable)
img_eq = cv2.equalizeHist(img)

# Expand to 3 channels for LIME
img_eq = np.stack([img_eq] * 3, axis=-1)
img_eq = img_eq.astype('double') / 255.0

# ==== 2. Define prediction function ====
def predict_fn(images):
    images = np.array(images, dtype=np.float32)
    return final_model.predict(images)

# ==== 3. Define segmentation function (optional) ====
# If you already have one, use it; otherwise, you can define a simple one:
from skimage.segmentation import slic
def segmentation_fn(image):
    return slic(image, n_segments=100, compactness=10, sigma=1)

# ==== 4. Run LIME explainer ====
explainer = lime_image.LimeImageExplainer()

explanation = explainer.explain_instance(
    image=img_eq,
    classifier_fn=predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=1500,
    segmentation_fn=segmentation_fn
)

# ==== 5. Get prediction class ====
pred_class = np.argmax(predict_fn(np.expand_dims(img_eq, axis=0)))

# ==== 6. Get LIME mask ====
temp, mask = explanation.get_image_and_mask(
    label=pred_class,
    positive_only=False,
    num_features=10,
    hide_rest=False
)

# ==== 7. Visualization: Original vs LIME ====
plt.figure(figsize=(10, 5))

# Original CT
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original CT Image")
plt.axis('off')

# LIME Explanation
plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(temp / np.max(temp), mask))
plt.title(f"LIME Explanation (Class {pred_class})")
plt.axis('off')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/LIME_Explanation_{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
plt.show()

#Final Prediction 
test_preds = final_model.predict(X_test)
y_test_pred = np.argmax(test_preds, axis=1)
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap=plt.cm.Blues)
plt.title("Final Test Confusion Matrix")
plt.savefig(f"{RESULTS_DIR}/final_test_confusion_matrix{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
plt.show()

#Final Metrics Summary
NUM_CLASSES = len(np.unique(y_all))  # or manually define if you already know it

y_temp_cat = tf.keras.utils.to_categorical(y_temp, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

train_preds = final_model.predict(X_temp)
train_labels = np.argmax(y_temp_cat, axis=1)
train_pred_labels = np.argmax(train_preds, axis=1)
test_pred_labels = np.argmax(test_preds, axis=1)

final_metrics = {
    "Dataset": ["Training", "Testing"],
    "Accuracy": [
        accuracy_score(train_labels, train_pred_labels),
        accuracy_score(y_test, test_pred_labels)
    ],
    "Precision": [
        precision_score(train_labels, train_pred_labels, average="macro"),
        precision_score(y_test, test_pred_labels, average="macro")
    ],
    "Recall": [
        recall_score(train_labels, train_pred_labels, average="macro"),
        recall_score(y_test, test_pred_labels, average="macro")
    ],
    "F1-score": [
        f1_score(train_labels, train_pred_labels, average="macro"),
        f1_score(y_test, test_pred_labels, average="macro")
    ]
}

df_final = pd.DataFrame(final_metrics)
print("\n=== Final Training vs Testing Metrics ===")
print(df_final.to_string(index=False))

# Accuracy & Loss Plots
train_loss = final_model.evaluate(X_temp, y_temp_cat, verbose=0)[0]
test_loss = final_model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, NUM_CLASSES), verbose=0)[0]

plt.figure()
plt.bar(["Train", "Test"], [train_loss, test_loss])
plt.title("Loss Comparison")
plt.ylabel("Loss")
plt.savefig(f"{RESULTS_DIR}/train_vs_test_loss.png")
plt.close()

plt.figure()
plt.bar(["Train", "Test"], df_final["Accuracy"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig(f"{RESULTS_DIR}/train_vs_test_accuracy{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
plt.close()
