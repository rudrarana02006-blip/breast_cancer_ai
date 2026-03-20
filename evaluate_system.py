import torch
import os
from src.engine import IndustryMedicalAI
from src.utils.data_manager import get_real_clinical_data
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    # Force MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔍 Validating HIGH-RES Model (30 Features) on: {str(device).upper()}")

    # 1. Initialize the UPGRADED architecture (Must match the 30 from training)
    model = IndustryMedicalAI(num_clinical_features=30).to(device)
    
    # 2. Load the Weights
    model_path = "models/breast_cancer_model.pth"
    if not os.path.exists(model_path):
        print("❌ Error: Trained model file not found!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Get the 30-feature Test Data
    _, X_test, _, y_test = get_real_clinical_data()
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 4. Predict
    print("🚀 Running Inference...")
    with torch.no_grad():
        # Create zero-tensors for the vision part (since we're testing clinical logic)
        fake_images = torch.zeros(X_test.size(0), 3, 224, 224).to(device)
        outputs = model(fake_images, X_test)
        _, predicted = torch.max(outputs, 1)

    # 5. Show Industry Metrics
    y_true = y_test.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    print("\n" + "="*45)
    print("📊 CLINICAL VALIDATION REPORT (30 FEATURES)")
    print("="*45)
    print(classification_report(y_true, y_pred, target_names=['Malignant', 'Benign']))
    
    print("\n🏥 CONFUSION MATRIX (Actual vs Predicted)")
    print(confusion_matrix(y_true, y_pred))
    print("="*45)

if __name__ == "__main__":
    evaluate()
