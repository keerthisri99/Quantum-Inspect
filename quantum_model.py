```python
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
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
# FILE PATHS
# ============================================================

DATA_X_PATH = "data/X.npy"
DATA_Y_PATH = "data/y.npy"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "qsvc_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "qsvc_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "qsvc_config.pkl")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    X = np.load(DATA_X_PATH)
    y = np.load(DATA_Y_PATH)

    print("Dataset loaded successfully!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Number of features:", X.shape[1])

    return X, y


# ============================================================
# CHECK WHETHER A TRAINED MODEL EXISTS
# ============================================================

def model_exists():
    return (
        os.path.isfile(MODEL_PATH)
        and os.path.isfile(SCALER_PATH)
        and os.path.isfile(CONFIG_PATH)
    )


# ============================================================
# CREATE QUANTUM MODEL
# ============================================================

def create_quantum_model(reps, C, feature_dimension):
    feature_map = zz_feature_map(
        feature_dimension=feature_dimension,
        reps=reps
    )

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    model = QSVC(
        quantum_kernel=quantum_kernel,
        C=C
    )

    return model


# ============================================================
# EVALUATE CONFIGURATION USING STRATIFIED CROSS-VALIDATION
# ============================================================

def evaluate_configuration(X_train, y_train, reps, C):

    _, class_counts = np.unique(
        y_train,
        return_counts=True
    )

    min_class_count = np.min(class_counts)
    n_splits = min(3, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "Each class needs at least 2 training samples "
            "for stratified cross-validation."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_index, val_index) in enumerate(
        skf.split(X_train, y_train), start=1
    ):

        print(f"    Cross-validation fold {fold}/{n_splits}")

        X_fold_train = X_train[train_index]
        X_fold_val = X_train[val_index]

        y_fold_train = y_train[train_index]
        y_fold_val = y_train[val_index]

        scaler = StandardScaler()

        X_fold_train_scaled = scaler.fit_transform(
            X_fold_train
        )

        X_fold_val_scaled = scaler.transform(
            X_fold_val
        )

        model = create_quantum_model(
            reps=reps,
            C=C,
            feature_dimension=X_train.shape[1]
        )

        model.fit(
            X_fold_train_scaled,
            y_fold_train
        )

        predictions = model.predict(
            X_fold_val_scaled
        )

        accuracy_scores.append(
            accuracy_score(y_fold_val, predictions)
        )

        precision_scores.append(
            precision_score(
                y_fold_val,
                predictions,
                zero_division=0
            )
        )

        recall_scores.append(
            recall_score(
                y_fold_val,
                predictions,
                zero_division=0
            )
        )

        f1_scores.append(
            f1_score(
                y_fold_val,
                predictions,
                zero_division=0
            )
        )

    return {
        "accuracy": np.mean(accuracy_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "f1": np.mean(f1_scores)
    }


# ============================================================
# FIND BEST QUANTUM CONFIGURATION
# ============================================================

def find_best_configuration(X_train, y_train):

    reps_values = [1]
    C_values = [0.1, 1]

    best_configuration = None
    best_f1 = -1

    print("\n")
    print("SEARCHING FOR BEST QUANTUM CONFIGURATION")
    print("Feature-map reps:", reps_values)
    print("QSVC C values   :", C_values)

    for reps in reps_values:
        for C in C_values:

            print(
                f"\nTesting ZZFeatureMap reps={reps}, "
                f"QSVC C={C}"
            )

            scores = evaluate_configuration(
                X_train,
                y_train,
                reps,
                C
            )

            print(f"  Accuracy : {scores['accuracy']:.4f}")
            print(f"  Precision: {scores['precision']:.4f}")
            print(f"  Recall   : {scores['recall']:.4f}")
            print(f"  F1       : {scores['f1']:.4f}")

            if scores["f1"] > best_f1:
                best_f1 = scores["f1"]

                best_configuration = {
                    "reps": reps,
                    "C": C,
                    "accuracy": scores["accuracy"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"]
                }

    print("\n" + "=" * 65)
    print("BEST CONFIGURATION")
    print("=" * 65)
    print(f"ZZFeatureMap reps : {best_configuration['reps']}")
    print(f"QSVC C            : {best_configuration['C']}")
    print(f"CV Accuracy       : {best_configuration['accuracy']:.4f}")
    print(f"CV Precision      : {best_configuration['precision']:.4f}")
    print(f"CV Recall         : {best_configuration['recall']:.4f}")
    print(f"CV F1             : {best_configuration['f1']:.4f}")
    print("=" * 65)

    return best_configuration


# ============================================================
# TRAIN AND SAVE MODEL
# ============================================================

def train_model():

    X, y = load_dataset()

    print("\n")
    print("=" * 65)
    print("QUANTUMINSPECT MODEL TRAINING")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDataset:")
    print("Total samples :", len(X))
    print("Training      :", len(X_train))
    print("Final testing :", len(X_test))

    best = find_best_configuration(
        X_train,
        y_train
    )

    best_reps = best["reps"]
    best_C = best["C"]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining final quantum model...")

    final_model = create_quantum_model(
        reps=best_reps,
        C=best_C,
        feature_dimension=X.shape[1]
    )

    final_model.fit(
        X_train_scaled,
        y_train
    )

    test_predictions = final_model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(y_test, test_predictions)

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    confusion = confusion_matrix(
        y_test,
        test_predictions
    )

    print("\n")
    print("=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)
    print(f"Best reps      : {best_reps}")
    print(f"Best C         : {best_C}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion)
    print("=" * 65)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the trained model
    joblib.dump(final_model, MODEL_PATH)

    # Save the fitted scaler
    joblib.dump(scaler, SCALER_PATH)

    # Save configuration and evaluation metrics
    joblib.dump(
        {
            "reps": best_reps,
            "C": best_C,
            "feature_dimension": X.shape[1],
            "cv_accuracy": best["accuracy"],
            "cv_precision": best["precision"],
            "cv_recall": best["recall"],
            "cv_f1": best["f1"],
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        },
        CONFIG_PATH
    )

    print("\nModel saved successfully!")
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(CONFIG_PATH)

    return final_model, scaler


# ============================================================
# LOAD PREVIOUSLY TRAINED MODEL
# ============================================================

def load_model():

    if not model_exists():
        raise FileNotFoundError(
            "A complete trained model was not found. "
            "Train the model first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = joblib.load(CONFIG_PATH)

    print("Previously trained model loaded successfully!")
    print("Training will be skipped.")
    print("Saved configuration:", config)

    return model, scaler


# ============================================================
# GET MODEL: LOAD IF AVAILABLE, OTHERWISE TRAIN
# ============================================================

def get_model():

    if model_exists():
        print("\nSaved model found. Loading without retraining...")
        return load_model()

    print("\nNo complete saved model found.")
    print("Training the model for the first time...")

    return train_model()


# ============================================================
# PREDICT GOOD OR DEFECTIVE
# ============================================================

def predict(features):

    # Loads the saved model; trains only if model files are missing.
    model, scaler = get_model()

    features = np.asarray(features, dtype=float)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise ValueError(
            "Features must be a 1D or 2D numeric array."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    expected_features = scaler.n_features_in_

    if features.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, "
            f"but received {features.shape[1]}."
        )

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    # Preserves your original label mapping.
    if prediction[0] == 0:
        return "GOOD"

    return "DEFECTIVE"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Running this file:
    # 1. Loads existing model files, if all are present.
    # 2. Otherwise, trains and saves the model.
    get_model()
```
