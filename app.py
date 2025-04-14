from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import joblib
from utils import extract_glcm_features, get_stage_from_glcm_features

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('lung_cancer_model.h5')
scaler = joblib.load('scaler.pkl')

# Set thresholds (hardcode from your training set)
contrast_qs = [0.02, 0.05, 0.08]
entropy_qs = [1.5, 2.5, 3.5]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(filepath)

    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (64, 64))
    img_norm = img_resized / 255.0
    img_input = img_norm.reshape(1, -1)
    img_input_scaled = scaler.transform(img_input)

    pred = model.predict(img_input_scaled)
    class_pred = np.argmax(pred)

    if class_pred == 2:
        result = "Prediction: Normal (No Cancer)"
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) / 255.0
        contrast, entropy = extract_glcm_features(gray)
        stage = get_stage_from_glcm_features(contrast, entropy, contrast_qs, entropy_qs, class_pred)
        result = f"Prediction: {'Benign' if class_pred == 0 else 'Malignant'} - {stage}"

    return render_template('index.html', prediction=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
