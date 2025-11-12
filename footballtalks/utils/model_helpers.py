# Helper for ML-related functions
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_demo_model():
    """Train a demo RandomForest model for win prediction."""
    X_train = np.array([[20,1,1],[15,1,0],[30,0,-1],[10,0,0]])
    y_train = np.array([1,0,1,0])  # 1=Home Win, 0=Away Win
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
