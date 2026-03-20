import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.engine import IndustryMedicalAI
from src.utils.data_manager import get_real_clinical_data
import os

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training starting on: {str(device).upper()}")

    # 1. Load Data (Now getting all 30 features)
    X_train, X_test, y_train, y_test = get_real_clinical_data()
    
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Initialize Model with 30 FEATURES
    model = IndustryMedicalAI(num_clinical_features=30).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🏋️ Training Loop Initiated (High-Resolution 30-Feature Mode)...")
    model.train()
    
    for epoch in range(20): # Increased to 20 for better precision
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            fake_images = torch.randn(data.size(0), 3, 224, 224).to(device)
            
            optimizer.zero_grad()
            outputs = model(fake_images, data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/20] | Avg Loss: {running_loss/len(train_loader):.4f}")

    if not os.path.exists('models'): os.makedirs('models')
    torch.save(model.state_dict(), "models/breast_cancer_model.pth")
    print("\n✅ Training Complete. High-Resolution weights saved!")

if __name__ == "__main__":
    train()
