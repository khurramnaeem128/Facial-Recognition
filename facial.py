
import os
import pandas as pd
import numpy as np
import cv2

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


dataset_path = "lfw_funneled"

people = os.listdir(dataset_path)

print("Number of people:", len(people))
print("First 10 people:", people[:10])


# 2. CHECK FIRST PERSON


first_person = people[0]

person_path = os.path.join(dataset_path, first_person)

images = os.listdir(person_path)

print("First person:", first_person)
print("Number of images:", len(images))
print("Images:", images[:5])



# 3. COUNT IMAGES


image_counts = []

for person in people:

    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        images = os.listdir(person_path)

        image_counts.append(len(images))


print("Total images:", sum(image_counts))
print("Maximum images for one person:", max(image_counts))




# 4. SELECT PEOPLE WITH AT LEAST 50 IMAGES

selected_people = []

for person in people:

    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        images = os.listdir(person_path)

        if len(images) >= 50:

            selected_people.append(person)


print("People with at least 10 images:", len(selected_people))

print("\nFirst 50 selected people and their image counts:")

for person in selected_people[:10]:

    person_path = os.path.join(dataset_path, person)

    images = os.listdir(person_path)

    print(person, "→", len(images), "images")


# 5. EXAMPLE IMAGE + HOG


first_person = selected_people[0]

person_path = os.path.join(dataset_path, first_person)
image_name = os.listdir(person_path)[0]
image_path = os.path.join(person_path, image_name)

image = cv2.imread(image_path)

print("\nPerson:", first_person)
print("Image:", image_name)
print("Image shape:", image.shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Gray image shape:", gray_image.shape)

resized_image = cv2.resize(gray_image, (50, 50))
print("Resized image shape:", resized_image.shape)

features = hog(
    resized_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2)
)

print("HOG features shape:", features.shape)
print("First 10 HOG features:", features[:10])


# 6. CREATE FEATURE DATASET

X = []
y = []


for person in selected_people:

    person_path = os.path.join(dataset_path, person)

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (50, 50))

        features = hog(
            resized_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        X.append(features)
        y.append(person)


# Convert to NumPy arrays

X = np.array(X)
y = np.array(y)


print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("First label:", y[0])
print("First image features:", X[0].shape)

# 7. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# 8. SVM MODEL

model = SVC(
    kernel="linear", C=0.1
)

print("\nModel created successfully")


# Train

model.fit(X_train, y_train)

print("Model training completed successfully")

# 9. PREDICTION

y_pred = model.predict(X_test)

# 10. ACTUAL VS PREDICTED

comparison = pd.DataFrame({
    "Actual": y_test[:10],
    "Predicted": y_pred[:10]
})

print("\nActual vs Predicted Table:")
print(comparison.to_string(index=False))

# 11. ACCURACY

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Facial Recognition Confusion Matrix")

plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()


# 18. FINAL SVM MODEL

final_model = SVC(
    kernel="poly",
    C=1
)


final_model.fit(X_train, y_train)
print("\nFinal model training completed")

final_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_pred)
print("Final Accuracy:", final_accuracy)
print("Final Accuracy %:", final_accuracy * 100)

# 19. PREDICT A NEW FACE

new_image_path = "new_face.jpg"

new_image = cv2.imread(new_image_path)

if new_image is None:
    print("Image not found!")

else:

    gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (50, 50))
    new_features = hog(
        resized_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )

    new_features = new_features.reshape(1, -1)
    prediction = final_model.predict(new_features)
    print("\nPredicted Person:", prediction[0])