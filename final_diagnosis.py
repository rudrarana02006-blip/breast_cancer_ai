import torch
import torch.nn as nn
import torchvision.models as models
import fitz  # PyMuPDF
import os

# 1. SET UP THE MAC GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. DEFINE THE BRAIN (Your AI Architecture)
class CombinedMedicalAI(nn.Module):
    def __init__(self):
        super(CombinedMedicalAI, self).__init__()
        # Vision Branch (ResNet50)
        self.vision = models.resnet50(weights='IMAGENET1K_V1')
        self.vision.fc = nn.Identity() 
        
        # Clinical Branch (Processes 4 keywords we find in the PDF)
        self.clinical = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        
        # Decision Maker (Vision + Clinical)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, img, clinical_data):
        v_feat = self.vision(img)
        c_feat = self.clinical(clinical_data)
        combined = torch.cat((v_feat, c_feat), dim=1)
        return self.classifier(combined)

# 3. SCAN PDF AND CONVERT TO NUMBERS (0 or 1)
def scan_report_to_numbers(pdf_path):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc]).lower()
    
    # Define our 4 key features
    keywords = ["malignant", "bi-rads", "mass", "calcification"]
    # If found = 1.0, if not = 0.0
    features = [1.0 if word in text else 0.0 for word in keywords]
    
    return torch.tensor([features]).to(device)

# 4. RUN THE FULL DIAGNOSIS
def run_full_system():
    # --- FIX IS HERE: str(device).upper() ---
    print(f"🚀 Initializing AI on {str(device).upper()}...")
    model = CombinedMedicalAI().to(device).eval()
    
    # Get the report data
    clinical_tensor = scan_report_to_numbers("test_report.pdf")
    
    # Simulate a mammogram scan
    fake_scan = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(fake_scan, clinical_tensor)
        probs = torch.softmax(output, dim=1)
        malignant_score = probs[0][1].item() * 100

    print("\n--- 🩺 FINAL AI DIAGNOSIS ---")
    print(f"Report Analysis: {clinical_tensor.cpu().tolist()[0]}")
    print(f"Malignancy Risk: {malignant_score:.2f}%")
    print(f"Recommendation: {'Urgent Biopsy' if malignant_score > 50 else 'Routine Follow-up'}")

if __name__ == "__main__":
    run_full_system()
