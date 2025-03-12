import pymysql
import base64
import io
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# ===================== STEP 1: FETCH IMAGES FROM MYSQL =====================
print("Fetching images from MySQL...")

try:
    # MySQL Connection (update your connection details)
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="root123",
        database="image_upload"
    )
    cursor = conn.cursor()

    # Fetch images and labels from your table
    cursor.execute("SELECT image_data, label FROM images1")
    data = cursor.fetchall()
except Exception as e:
    print(f"Error connecting to MySQL or fetching data: {e}")
    exit(1)
finally:
    if conn:
        conn.close()

# Decode Base64 images and collect labels
images, labels = [], []
for base64_string, label in data:
    try:
        base64_string = base64_string.split(",")[-1]  # Removes any prefix like "data:image/jpeg;base64,"
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        images.append(np.array(image))
        labels.append(label)
    except Exception as e:
        print(f"Error decoding image: {e}")

print(f"Fetched {len(images)} images.")

# ===================== STEP 2: PREPROCESS IMAGES =====================
print("Preprocessing images...")

# Define image size and resize images
IMG_SIZE = (64, 64)
images_resized = []
for img in images:
    if img.size > 0:  # Ensure the image is not empty
        img_resized = tf.image.resize(img, IMG_SIZE)
        images_resized.append(img_resized)
    else:
        print("Warning: Skipping an empty or invalid image.")
images_resized = np.array(images_resized) / 255.0  # Normalize pixel values

# Encode labels (assumes your labels are strings)
unique_labels = list(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels_encoded = np.array([label_map[label] for label in labels])
labels_one_hot = to_categorical(labels_encoded, num_classes=len(unique_labels))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images_resized, labels_one_hot, test_size=0.4, random_state=42
)

print(f"Training set: {len(X_train)} images, Testing set: {len(X_test)} images.")

# ===================== STEP 3: BUILD & TRAIN CNN MODEL =====================
print("Building CNN model...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with callbacks
print("Training model...")
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3)
]
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), callbacks=callbacks)

# ===================== STEP 4: SAVE MODEL IN JSON FORMAT =====================
print("Saving model in JSON format...")

# Save the model architecture to JSON
model_json = model.to_json()
with open("C:/hand/hand_sign_model.json", "w") as json_file:
    json_file.write(model_json)
print("Model architecture saved to JSON.")

# Save the model weights
model.save_weights("C:/hand/hand_sign_model_weights.h5")
print("Model weights saved.")

# ===================== STEP 5: REAL-TIME HAND SIGN DETECTION =====================
print("Starting real-time hand sign detection...")

# Load the model architecture from JSON
with open("C:/hand/hand_sign_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the weights into the model
loaded_model.load_weights("C:/hand/hand_sign_model_weights.h5")
print("Model loaded from JSON and weights.")

# Compile the loaded model
loaded_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Start webcam for real-time detection
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get bounding box coordinates
            h, w, _ = frame.shape
            x_min = min([int(l.x * w) for l in hand_landmarks.landmark])
            y_min = min([int(l.y * h) for l in hand_landmarks.landmark])
            x_max = max([int(l.x * w) for l in hand_landmarks.landmark])
            y_max = max([int(l.y * h) for l in hand_landmarks.landmark])
            
            # Ensure valid bounding box
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # Crop and preprocess hand region for prediction
            if x_max > x_min and y_max > y_min:  # Ensure non-empty crop
                hand_img = frame[y_min:y_max, x_min:x_max]
                
                if hand_img.size > 0:  # Ensure valid image before resizing
                    hand_img = cv2.resize(hand_img, IMG_SIZE)
                    hand_img = np.expand_dims(hand_img / 255.0, axis=0)

                    # Predict the hand sign
                    prediction = loaded_model.predict(hand_img)
                    predicted_index = np.argmax(prediction)
                    predicted_label = unique_labels[predicted_index]

                    # Display prediction on screen
                    cv2.putText(frame, predicted_label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Hand Sign Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")