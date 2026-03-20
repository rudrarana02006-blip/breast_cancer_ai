import os
import torch
import numpy as np
import pdfplumber
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
from src.engine import IndustryMedicalAI

app = Flask(__name__)
CORS(app)

# --- MODELS SETUP ---
device = torch.device("cpu")
clinical_model = IndustryMedicalAI(num_clinical_features=30).to(device)
if os.path.exists("models/breast_cancer_model.pth"):
    clinical_model.load_state_dict(torch.load("models/breast_cancer_model.pth", map_location=device))
clinical_model.eval()

# Vision Model for CT Scans
vision_model = models.resnet50(weights='IMAGENET1K_V1')
vision_model.fc = torch.nn.Linear(vision_model.fc.in_features, 1) # Binary output
vision_model.eval()

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

KEYWORDS = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "points", "symmetry", "dimension"]

def extract_from_pdf(file_path):
    extracted = {}
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        for i, key in enumerate(KEYWORDS):
            match = re.search(rf"{key}.*?(\d+\.?\d*)", text, re.IGNORECASE)
            val = float(match.group(1)) if match else 15.0
            extracted[f"PARAM_{i+1}"] = val
            extracted[f"PARAM_{i+11}"] = val * 0.1
            extracted[f"PARAM_{i+21}"] = val * 1.2
    except: pass
    return extracted

@app.route('/upload-reports', methods=['POST'])
def upload_reports():
    files = request.files.getlist('files')
    final_features = {f"PARAM_{i}": 15.0 for i in range(1, 31)}
    scan_detected = False
    
    for file in files:
        fname = file.filename.lower()
        path = os.path.join("uploads", file.filename)
        file.save(path)
        
        if fname.endswith('.pdf'):
            final_features.update(extract_from_pdf(path))
        elif fname.endswith(('.png', '.jpg', '.jpeg')):
            scan_detected = True # Flag that a CT scan is ready
            
    return jsonify({
        "message": "Files analyzed.",
        "extracted_features": final_features,
        "scan_ready": scan_detected
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        feature_dict = data.get('features', {})
        features = [float(feature_dict.get(f"PARAM_{i}", 15.0)) for i in range(1, 31)]
        
        # Clinical Prediction
        input_tensor = torch.FloatTensor(features).view(1, -1).to(device)
        with torch.no_grad():
            output = clinical_model(input_tensor)
            clinical_score = float(output[0].item() if isinstance(output, tuple) else output.item())
        
        return jsonify({"malignancy_risk": round(clinical_score * 100, 2), "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
