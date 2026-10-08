
import os
import joblib
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

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



X = np.load("data/X.npy")
y = np.load("data/y.npy")

print("\n===================================")
print("          DATA INFORMATION")
print("===================================")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nClass distribution:")
print(np.unique(y, return_counts=True))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\n===================================")
print("          DATA SPLIT")
print("===================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


print("\n===================================")
print("      MUTUAL INFORMATION")
print("===================================\n")

# MI is calculated ONLY on training data
mi_scores = mutual_info_classif(
    X_train,
    y_train,
    random_state=42
)

for i, score in enumerate(mi_scores):
    print(
        f"Feature {i + 1}: {score:.4f}"
    )


candidate_count = min(
    6,
    X_train.shape[1]
)

candidate_features = np.argsort(
    mi_scores
)[-candidate_count:]

candidate_features = np.sort(
    candidate_features
)

print("\nCandidate features:")
print(candidate_features + 1)


def create_quantum_kernel(
    number_of_features,
    reps=2
):

    feature_map = zz_feature_map(
        feature_dimension=number_of_features,
        reps=reps
    )

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    return quantum_kernel


print("\n===================================")
print("     SEARCHING FEATURE SUBSETS")
print("===================================")

feature_sizes = [
    2,
    3,
    4
]

# Keep this small because quantum kernels
# are computationally expensive.
C_values = [
    0.1,
    1.0,
    10.0
]

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

best_score = -1
best_features = None
best_C = None
best_reps = None


for number_of_features in feature_sizes:

    if number_of_features > len(candidate_features):
        continue

    # Take the highest MI-ranked features
    selected_features = candidate_features[
        -number_of_features:
    ]

    print(
        f"\nTesting {number_of_features} features:"
    )

    print(
        "Features:",
        selected_features + 1
    )


    # Select features
    X_train_subset = X_train[
        :,
        selected_features
    ]


    # Try small number of repetitions
    for reps in [1, 2]:

        print(
            f"  Feature-map reps = {reps}"
        )


        # Create quantum kernel
        quantum_kernel = create_quantum_kernel(
            number_of_features,
            reps
        )


        # Try different C values
        for C in C_values:

            print(
                f"    Testing C = {C}"
            )


            scaler = StandardScaler()


            # QSVC
            model = QSVC(
                quantum_kernel=quantum_kernel,
                C=C
            )


            # ------------------------------------------------
            # Manual CV
            # ------------------------------------------------

            fold_scores = []


            for train_idx, val_idx in cv.split(
                X_train_subset,
                y_train
            ):

                X_fold_train = (
                    X_train_subset[train_idx]
                )

                X_fold_val = (
                    X_train_subset[val_idx]
                )

                y_fold_train = (
                    y_train[train_idx]
                )

                y_fold_val = (
                    y_train[val_idx]
                )


                # Scale using ONLY fold training data
                scaler_fold = StandardScaler()

                X_fold_train_scaled = (
                    scaler_fold.fit_transform(
                        X_fold_train
                    )
                )

                X_fold_val_scaled = (
                    scaler_fold.transform(
                        X_fold_val
                    )
                )


                # Train
                model.fit(
                    X_fold_train_scaled,
                    y_fold_train
                )


                # Validate
                predictions = model.predict(
                    X_fold_val_scaled
                )


                score = accuracy_score(
                    y_fold_val,
                    predictions
                )


                fold_scores.append(score)


            mean_score = np.mean(
                fold_scores
            )


            print(
                f"      CV Accuracy = "
                f"{mean_score:.4f}"
            )


            # ------------------------------------------------
            # Save best configuration
            # ------------------------------------------------

            if mean_score > best_score:

                best_score = mean_score

                best_features = (
                    selected_features.copy()
                )

                best_C = C
                best_reps = reps


# ============================================================
# 7. DISPLAY BEST CONFIGURATION
# ============================================================

print("\n===================================")
print("       BEST CONFIGURATION")
print("===================================")

print(
    "Best CV Accuracy:",
    f"{best_score:.4f}"
)

print(
    "Best features:",
    best_features + 1
)

print(
    "Best C:",
    best_C
)

print(
    "Best reps:",
    best_reps
)


# ============================================================
# 8. FINAL TRAINING
# ============================================================

print("\n===================================")
print("        FINAL TRAINING")
print("===================================")


# Select best features
X_train_final = X_train[
    :,
    best_features
]

X_test_final = X_test[
    :,
    best_features
]


# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_final
)

X_test_scaled = scaler.transform(
    X_test_final
)


# ------------------------------------------------------------
# Final quantum kernel
# ------------------------------------------------------------

final_feature_map = zz_feature_map(
    feature_dimension=len(best_features),
    reps=best_reps
)

final_quantum_kernel = FidelityQuantumKernel(
    feature_map=final_feature_map
)


# ------------------------------------------------------------
# Final QSVC
# ------------------------------------------------------------

final_model = QSVC(
    quantum_kernel=final_quantum_kernel,
    C=best_C
)


print("\nTraining final QSVC...")

final_model.fit(
    X_train_scaled,
    y_train
)

print("Final QSVC training completed!")


# ============================================================
# 9. FINAL TEST
# ============================================================

predictions = final_model.predict(
    X_test_scaled
)


# ============================================================
# 10. METRICS
# ============================================================

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


# ============================================================
# 11. RESULTS
# ============================================================

print("\n===================================")
print("       FINAL MODEL PERFORMANCE")
print("===================================\n")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print("\nConfusion Matrix:")
print(confusion)


# ============================================================
# 12. SAVE EVERYTHING
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    final_model,
    "model/qsvc_model.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)

joblib.dump(
    best_features,
    "model/selected_features.pkl"
)

joblib.dump(
    mi_scores,
    "model/mi_scores.pkl"
)

joblib.dump(
    {
        "C": best_C,
        "reps": best_reps,
        "cv_accuracy": best_score
    },
    "model/config.pkl"
)


print("\n===================================")
print("          MODEL SAVED")
print("===================================")

print("model/qsvc_model.pkl")
print("model/scaler.pkl")
print("model/selected_features.pkl")
print("model/mi_scores.pkl")
print("model/config.pkl")


# ============================================================
# 13. LOAD MODEL
# ============================================================

def load_model():

    model = joblib.load(
        "model/qsvc_model.pkl"
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    selected_features = joblib.load(
        "model/selected_features.pkl"
    )

    return (
        model,
        scaler,
        selected_features
    )


# ============================================================
# 14. PREDICT NEW SAMPLE
# ============================================================

def predict(features):

    model, scaler, selected_features = (
        load_model()
    )


    # Convert to numpy
    features = np.asarray(
        features
    )


    # Make 2D
    if features.ndim == 1:

        features = features.reshape(
            1,
            -1
        )


    # Select the SAME features
    # used during training
    features_selected = features[
        :,
        selected_features
    ]


    # Scale
    features_scaled = scaler.transform(
        features_selected
    )


    # Predict
    prediction = model.predict(
        features_scaled
    )


    if prediction[0] == 0:

        return "GOOD"

    else:

        return "DEFECTIVE"


# ============================================================
# 15. END
# ============================================================

if __name__ == "__main__":

    print(
        "\nQuantumInspect training completed."
    )
