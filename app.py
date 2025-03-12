
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import io
import base64
import cv2
import mediapipe as mp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the hand sign model
with open("hand_sign_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("C:/Users/DELL/Desktop/hand_sign_detection/hand_sign_model.h5")
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Unique labels
unique_labels = ["Hello", "I Love You", "Thank You", "No"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)#updated


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Invalid request, no image data received"}), 400
        
        # Process base64 image
        image_data = data['image'].split(",")[1]  # Remove "data:image/jpeg;base64," prefix
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        
        # Convert to OpenCV format
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        print("Hand Detection Results:", results.multi_hand_landmarks)#updated

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box coordinates
                h, w, _ = frame.shape
                x_min = min([int(l.x * w) for l in hand_landmarks.landmark])
                y_min = min([int(l.y * h) for l in hand_landmarks.landmark])
                x_max = max([int(l.x * w) for l in hand_landmarks.landmark])
                y_max = max([int(l.y * h) for l in hand_landmarks.landmark])
                
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                # Crop and preprocess for prediction
                if x_max > x_min and y_max > y_min:
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = np.expand_dims(hand_img / 255.0, axis=0)

                    # Make prediction
                    predictions = loaded_model.predict(hand_img)
                    predicted_label = unique_labels[np.argmax(predictions)]
                    
                    return jsonify({"predicted_label": predicted_label})
                
        return jsonify({"error": "No hand detected"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
