import streamlit as st
import torch
import os
from src.engine import IndustryMedicalAI

st.set_page_config(page_title="OncoVision AI", page_icon="🎗️")

# The fix for the error in your screenshot:
st.markdown("<style>.main { background-color: #f5f7f9; }</style>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = IndustryMedicalAI(num_clinical_features=30).to(device)
    if os.path.exists("models/breast_cancer_model.pth"):
        model.load_state_dict(torch.load("models/breast_cancer_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()
st.title("OncoVision AI Dashboard")
st.write("System is ready for clinical input.")
