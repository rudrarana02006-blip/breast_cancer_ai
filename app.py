import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.engine import IndustryMedicalAI

app = Flask(__name__)
CORS(app)

# --- MODEL SETUP ---
device = torch.device("cpu")
model = IndustryMedicalAI(num_clinical_features=30).to(device)
if os.path.exists("models/breast_cancer_model.pth"):
    model.load_state_dict(torch.load("models/breast_cancer_model.pth", map_location=device))
model.eval()

@app.route('/upload-reports', methods=['POST'])
def upload_reports():
    files = request.files.getlist('files')
    # This prepares the 30 features for the UI to display and fine-tune
    extracted_data = {}
    for i in range(1, 31):
        extracted_data[f"PARAM_{i}"] = 15.0 # Baseline placeholder
    
    return jsonify({
        "message": f"Successfully loaded {len(files)} files. Review parameters below.",
        "extracted_features": extracted_data
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        feature_dict = data.get('features', {})
        features = [float(val) for val in feature_dict.values()]
        
        if len(features) != 30:
            return jsonify({"error": f"Expected 30 features, got {len(features)}"}), 400

        input_tensor = torch.FloatTensor(features).view(1, -1).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_value = output[0].item() if isinstance(output, tuple) else output.item()
            prediction = round(float(pred_value) * 100, 2)
        
        return jsonify({"malignancy_risk": prediction, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
