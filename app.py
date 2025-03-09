from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "ser_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotion labels
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']

# Function to preprocess audio
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=(0, -1))  # Reshape for model input
    return mel_spec_db

# Route for the frontend
@app.route('/')
def index():
    return render_template('index.html')

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = "temp.wav"
    file.save(file_path)

    # Preprocess the audio file
    input_data = preprocess_audio(file_path)

    # Make a prediction
    prediction = model.predict(input_data)
    emotion = np.argmax(prediction)
    predicted_emotion = emotion_labels[emotion]

    return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)
