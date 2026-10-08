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


# Load features extracted by quant_hack.py
X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("Dataset loaded successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of features:", X.shape[1])
def create_quantum_model(reps, C):

    feature_map = zz_feature_map(
        feature_dimension=X.shape[1],
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


def evaluate_configuration(X_train, y_train, reps, C):

    
    class_counts = np.bincount(y_train)
    min_class_count = np.min(class_counts)

    n_splits = min(3, min_class_count)

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
            C=C
        )

        model.fit(
            X_fold_train_scaled,
            y_fold_train
        )


        
        predictions = model.predict(
            X_fold_val_scaled
        )

        accuracy = accuracy_score(
            y_fold_val,
            predictions
        )

        precision = precision_score(
            y_fold_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_fold_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_fold_val,
            predictions,
            zero_division=0
        )


        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)


    return {
        "accuracy": np.mean(accuracy_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "f1": np.mean(f1_scores)
    }

def find_best_configuration(X_train, y_train):

    reps_values = [1]

    C_values = [0.1, 1]


    best_configuration = None

    best_f1 = -1


    print("\n")
    
    print("SEARCHING FOR BEST QUANTUM CONFIGURATION")
    

    print("\nTesting:")

    print("Feature-map reps:", reps_values)
    print("QSVC C values   :", C_values)

    print("\n")

    for reps in reps_values:

        for C in C_values:

            print(
                f"Testing ZZFeatureMap reps={reps}, "
                f"QSVC C={C}"
            )

            scores = evaluate_configuration(
                X_train,
                y_train,
                reps,
                C
            )


            print(
                f"  Accuracy : {scores['accuracy']:.4f}"
            )

            print(
                f"  Precision: {scores['precision']:.4f}"
            )

            print(
                f"  Recall   : {scores['recall']:.4f}"
            )

            print(
                f"  F1       : {scores['f1']:.4f}"
            )

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


    print("\n")
    print("=" * 65)
    print("BEST CONFIGURATION")
    print("=" * 65)

    print(
        f"ZZFeatureMap reps : "
        f"{best_configuration['reps']}"
    )

    print(
        f"QSVC C            : "
        f"{best_configuration['C']}"
    )

    print(
        f"CV Accuracy       : "
        f"{best_configuration['accuracy']:.4f}"
    )

    print(
        f"CV Precision      : "
        f"{best_configuration['precision']:.4f}"
    )

    print(
        f"CV Recall         : "
        f"{best_configuration['recall']:.4f}"
    )

    print(
        f"CV F1             : "
        f"{best_configuration['f1']:.4f}"
    )

    print("=" * 65)

    return best_configuration


def train_model():

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

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("\nTraining final quantum model...")

    final_model = create_quantum_model(
        reps=best_reps,
        C=best_C
    )

    final_model.fit(
        X_train_scaled,
        y_train
    )


    test_predictions = final_model.predict(
        X_test_scaled
    )


    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

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

    print(
        f"Best reps      : {best_reps}"
    )

    print(
        f"Best C         : {best_C}"
    )

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(confusion)

    print("=" * 65)

    os.makedirs(
    "models",
    exist_ok=True
    )


    joblib.dump(
    final_model,
    "models/qsvc_model.pkl"
    )

    joblib.dump(
    scaler,
        "models/qsvc_scaler.pkl"
)

    # Save configuration too
    joblib.dump(
    {
        "reps": best_reps,
        "C": best_C,
        "cv_accuracy": best["accuracy"],
        "cv_precision": best["precision"],
        "cv_recall": best["recall"],
        "cv_f1": best["f1"]
    },
    "models/qsvc_config.pkl"
    )


    print("\nModel saved successfully!")

    print("models/qsvc_model.pkl")
    print("models/qsvc_scaler.pkl")
    print("models/qsvc_config.pkl")

def load_model():

    model = joblib.load(
        "models/qsvc_model.pkl"
    )

    scaler = joblib.load(
        "models/qsvc_scaler.pkl"
    )

    return model, scaler

def predict(features):

    model, scaler = load_model()


    features = np.array(features)



    if features.ndim == 1:
        features = features.reshape(1, -1)


    features_scaled = scaler.transform(
        features
    )

    prediction = model.predict(
        features_scaled
    )


    if prediction[0] == 0:
        return "GOOD"

    else:
        return "DEFECTIVE"
    
if __name__ == "__main__":

    train_model()
