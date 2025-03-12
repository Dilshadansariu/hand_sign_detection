from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the model
with open("C:/hand/hand_sign_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("C:/hand/hand_sign_model_weights.h5")
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# List of unique labels
unique_labels = ["Hello", "I Love You", "Thank You", "NO", "OK"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image'].split(",")[1]  # Remove the "data:image/jpeg;base64," prefix
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    
    # Preprocess the image
    image = image.resize((64, 64))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = loaded_model.predict(image)
    predicted_label = unique_labels[np.argmax(predictions)]

    # Return the result
    return jsonify({"predicted_label": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)