
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

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
# 1. FILE PATHS
# ============================================================

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR, "qsvc_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR, "scaler.pkl"
)

SELECTED_FEATURES_PATH = os.path.join(
    MODEL_DIR, "selected_features.pkl"
)

MI_SCORES_PATH = os.path.join(
    MODEL_DIR, "mi_scores.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR, "config.pkl"
)


# ============================================================
# 2. CHECK WHETHER ALL FIVE PICKLE FILES EXIST
# ============================================================

def model_exists():

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        SELECTED_FEATURES_PATH,
        MI_SCORES_PATH,
        CONFIG_PATH
    ]

    return all(
        os.path.isfile(path)
        for path in required_files
    )


# ============================================================
# 3. CREATE QUANTUM KERNEL
# ============================================================

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


# ============================================================
# 4. TRAIN MODEL AND SAVE ALL FIVE PICKLE FILES
# ============================================================

def train_model():

    # Load data only when training is required
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    print("\n===================================")
    print("       QUANTUMINSPECT TRAINING")
    print("===================================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nClass distribution:")
    print(np.unique(y, return_counts=True))

    # --------------------------------------------------------
    # DATA SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # --------------------------------------------------------
    # MUTUAL INFORMATION FEATURE SELECTION
    # --------------------------------------------------------

    print("\n===================================")
    print("       MUTUAL INFORMATION")
    print("===================================")

    # Calculate MI using training data only
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

    # --------------------------------------------------------
    # CONFIGURATION SEARCH
    # --------------------------------------------------------

    print("\n===================================")
    print("     SEARCHING FEATURE SUBSETS")
    print("===================================")

    feature_sizes = [2, 3, 4]
    C_values = [0.1, 1.0, 10.0]

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

        # Highest MI-ranked features
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

        X_train_subset = X_train[
            :,
            selected_features
        ]

        for reps in [1, 2]:

            print(
                f"  Feature-map reps = {reps}"
            )

            quantum_kernel = create_quantum_kernel(
                number_of_features,
                reps
            )

            for C in C_values:

                print(
                    f"    Testing C = {C}"
                )

                fold_scores = []

                # --------------------------------------------
                # STRATIFIED CROSS-VALIDATION
                # --------------------------------------------

                for fold, (train_idx, val_idx) in enumerate(
                    cv.split(X_train_subset, y_train),
                    start=1
                ):

                    print(
                        f"      Fold {fold}/3"
                    )

                    X_fold_train = X_train_subset[
                        train_idx
                    ]

                    X_fold_val = X_train_subset[
                        val_idx
                    ]

                    y_fold_train = y_train[
                        train_idx
                    ]

                    y_fold_val = y_train[
                        val_idx
                    ]

                    # Fit scaler only on fold training data
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

                    model = QSVC(
                        quantum_kernel=quantum_kernel,
                        C=C
                    )

                    model.fit(
                        X_fold_train_scaled,
                        y_fold_train
                    )

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
                    f"      CV Accuracy = {mean_score:.4f}"
                )

                # --------------------------------------------
                # RECORD BEST CONFIGURATION
                # --------------------------------------------

                if mean_score > best_score:

                    best_score = mean_score

                    best_features = (
                        selected_features.copy()
                    )

                    best_C = C
                    best_reps = reps

    if best_features is None:
        raise ValueError(
            "Unable to select features. "
            "Check the number of input features."
        )

    # --------------------------------------------------------
    # DISPLAY BEST CONFIGURATION
    # --------------------------------------------------------

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

    print("Best C:", best_C)
    print("Best reps:", best_reps)

    # --------------------------------------------------------
    # FINAL TRAINING
    # --------------------------------------------------------

    print("\n===================================")
    print("        FINAL TRAINING")
    print("===================================")

    X_train_final = X_train[
        :,
        best_features
    ]

    X_test_final = X_test[
        :,
        best_features
    ]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_final
    )

    X_test_scaled = scaler.transform(
        X_test_final
    )

    final_feature_map = zz_feature_map(
        feature_dimension=len(best_features),
        reps=best_reps
    )

    final_quantum_kernel = FidelityQuantumKernel(
        feature_map=final_feature_map
    )

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

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    predictions = final_model.predict(
        X_test_scaled
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    print("\n===================================")
    print("       FINAL MODEL PERFORMANCE")
    print("===================================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)

    # --------------------------------------------------------
    # SAVE ALL FIVE PICKLE FILES
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        best_features,
        SELECTED_FEATURES_PATH
    )

    joblib.dump(
        mi_scores,
        MI_SCORES_PATH
    )

    joblib.dump(
        {
            "C": best_C,
            "reps": best_reps,
            "cv_accuracy": best_score,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "number_of_selected_features": len(
                best_features
            ),
            "original_feature_count": X.shape[1]
        },
        CONFIG_PATH
    )

    print("\n===================================")
    print("          MODEL SAVED")
    print("===================================")

    print(MODEL_PATH)
    print(SCALER_PATH)
    print(SELECTED_FEATURES_PATH)
    print(MI_SCORES_PATH)
    print(CONFIG_PATH)

    print("\nTraining completed successfully!")


# ============================================================
# 5. LOAD PREVIOUSLY TRAINED MODEL
# ============================================================

def load_model():

    if not model_exists():
        raise FileNotFoundError(
            "One or more model files are missing."
        )

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    selected_features = joblib.load(
        SELECTED_FEATURES_PATH
    )

    # Load these files to verify they are available
    mi_scores = joblib.load(
        MI_SCORES_PATH
    )

    config = joblib.load(
        CONFIG_PATH
    )

    print("\nSaved model loaded successfully!")
    print("Training skipped.")
    print("Selected features:", selected_features + 1)
    print("Saved configuration:", config)

    return (
        model,
        scaler,
        selected_features
    )


# ============================================================
# 6. LOAD MODEL OR TRAIN IF NECESSARY
# ============================================================

def get_model():

    if model_exists():

        print("\nExisting trained model found.")
        print("Loading saved files without retraining.")

        return load_model()

    print("\nTrained model files not found or incomplete.")
    print("Training the model now...")

    train_model()

    # Load the newly saved model
    return load_model()


# ============================================================
# 7. PREDICT A NEW SAMPLE WITHOUT RETRAINING
# ============================================================

def predict(features):

    model, scaler, selected_features = get_model()

    features = np.asarray(
        features,
        dtype=float
    )

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise ValueError(
            "Features must be a 1D or 2D array."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    if features.shape[1] <= np.max(selected_features):
        raise ValueError(
            "Input has fewer features than required "
            "by the saved feature selector."
        )

    # Select exactly the features used during training
    features_selected = features[
        :,
        selected_features
    ]

    # Reuse the saved scaler
    features_scaled = scaler.transform(
        features_selected
    )

    # Predict using the saved model
    prediction = model.predict(
        features_scaled
    )

    if prediction[0] == 0:
        return "GOOD"

    return "DEFECTIVE"


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    if model_exists():

        print("\n===================================")
        print("      EXISTING MODEL DETECTED")
        print("===================================")

        print("Loading model...")
        get_model()

        print("\nReady for predictions.")
        print("No retraining performed.")

    else:

        print("\n===================================")
        print("       FIRST-TIME TRAINING")
        print("===================================")

        get_model()

        print("\nAll five pickle files are saved.")
        print("The model is ready for future predictions.")
