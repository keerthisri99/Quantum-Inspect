import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# ============================================================
# 1. TRAINING DATA
# ============================================================

# Features:
# 1. Mean intensity
# 2. Standard deviation
# 3. Edge density
# 4. Texture

X = np.load("data/X.npy")
y = np.load("data/y.npy")
# 1 = DEFECTIVE

y = np.array(
    [0] * 50 +
    [1] * 50
)
# ============================================================
# 2. TRAIN THE QUANTUM MODEL
# ============================================================

def train_model():

    print("\n===================================")
    print("       TRAINING QUANTUM MODEL")
    print("===================================\n")

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))


    # --------------------------------------------------------
    # Feature Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # --------------------------------------------------------
    # Quantum Feature Map
    # --------------------------------------------------------

    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=2
    )


    # --------------------------------------------------------
    # Quantum Kernel
    # --------------------------------------------------------

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )


    # --------------------------------------------------------
    # QSVC
    # --------------------------------------------------------

    model = QSVC(
        quantum_kernel=quantum_kernel
    )


    # --------------------------------------------------------
    # Train QSVC
    # --------------------------------------------------------

    print("\nTraining QSVC...")

    model.fit(
        X_train_scaled,
        y_train
    )

    print("QSVC training completed!")


    # ========================================================
    # 3. TEST THE MODEL
    # ========================================================

    predictions = model.predict(X_test_scaled)


    # ========================================================
    # 4. CALCULATE METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        predictions
    )


    # ========================================================
    # 5. DISPLAY RESULTS
    # ========================================================

    print("\n===================================")
    print("       MODEL PERFORMANCE")
    print("===================================\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)


    # ========================================================
    # 6. SAVE MODEL AND SCALER
    # ========================================================

    os.makedirs("model", exist_ok=True)

    joblib.dump(
        model,
        "model/qsvc_model.pkl"
    )

    joblib.dump(
        scaler,
        "model/scaler.pkl"
    )

    print("\n===================================")
    print("       MODEL SAVED")
    print("===================================")

    print("\nSaved files:")
    print("model/qsvc_model.pkl")
    print("model/scaler.pkl")


# ============================================================
# 7. LOAD SAVED MODEL
# ============================================================

def load_model():

    model = joblib.load(
        "model/qsvc_model.pkl"
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    return model, scaler


# ============================================================
# 8. PREDICT A NEW SAMPLE
# ============================================================

def predict(features):

    # Load already-trained model
    model, scaler = load_model()

    # Scale new features using
    # the SAME scaler used during training
    features_scaled = scaler.transform(
        features
    )

    # Make prediction
    prediction = model.predict(
        features_scaled
    )

    # Convert 0/1 to meaningful result

    if prediction[0] == 0:
        return "GOOD"

    else:
        return "DEFECTIVE"


# ============================================================
# 9. RUN TRAINING WHEN THIS FILE IS EXECUTED
# ============================================================

if __name__ == "__main__":

    train_model()