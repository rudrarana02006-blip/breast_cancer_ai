from sklearn.datasets import load_breast_cancer
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_real_clinical_data():
    data = load_breast_cancer()
    X = data.data # Use ALL 30 features now
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (torch.FloatTensor(X_train), torch.FloatTensor(X_test), 
            torch.LongTensor(y_train), torch.LongTensor(y_test))
