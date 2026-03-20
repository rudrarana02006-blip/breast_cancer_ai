import torch
import torch.nn as nn
from torchvision import models

class IndustryMedicalAI(nn.Module):
    def __init__(self, num_clinical_features=30): # Upgraded from 4 to 30
        super(IndustryMedicalAI, self).__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.clinical_branch = nn.Sequential(
            nn.Linear(num_clinical_features, 128), # Wider layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, img, clinical_data):
        v_feat = torch.flatten(self.vision_backbone(img), 1)
        c_feat = self.clinical_branch(clinical_data)
        return self.classifier(torch.cat((v_feat, c_feat), dim=1))
