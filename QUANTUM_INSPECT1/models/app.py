import streamlit as st
import cv2
import numpy as np
import joblib
from pathlib import Path

from src.feature_extraction import extract_features, FEATURE_NAMES


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="QuantumInspect",
    page_icon="⚛️",
    layout="wide"
)


# ============================================================
# LOAD MODELS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# Classical SVM
svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
svm_scaler = joblib.load( MODELS_DIR / "svm_scaler.pkl")
svm_metrics = joblib.load(MODELS_DIR / "svm_metrics.pkl")

# Quantum QSVC
qsvc_model = joblib.load(MODELS_DIR / "qsvc_model.pkl")
qsvc_scaler = joblib.load(MODELS_DIR / "qsvc_scaler.pkl")
qsvc_config = joblib.load( MODELS_DIR / "qsvc_config.pkl")


# ============================================================
# HEADER
# ============================================================

st.title("⚛️ QuantumInspect")

st.subheader("Quantum-Powered PCB Defect Inspection")

st.write(
    "Upload a PCB image to analyze it using our "
    "quantum machine learning inspection pipeline."
)


# ============================================================
# SYSTEM STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.success("🟢 System Online")

with col2:
    st.info("⚛️ Quantum Engine Ready")

with col3:
    st.success("🔍 Inspection Ready")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.divider()

st.subheader("📷 Upload PCB Image")

uploaded_file = st.file_uploader(
    "Upload a PCB image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# INSPECTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # --------------------------------------------------------
    # DISPLAY UPLOADED PCB
    # --------------------------------------------------------

    st.image(
        image,
        caption="Uploaded PCB",
        use_container_width=True
    )


    # ========================================================
    # INSPECT BUTTON
    # ========================================================

    if st.button("🔍 Inspect PCB"):

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        features = extract_features(image)

        features_2d = features.reshape(1, -1)


        # ----------------------------------------------------
        # CLASSICAL SVM PREDICTION
        # ----------------------------------------------------

        svm_features_scaled = svm_scaler.transform(
            features_2d
        )

        svm_prediction = svm_model.predict(
            svm_features_scaled
        )[0]

        if svm_prediction == 0:
            svm_result = "GOOD"
        else:
            svm_result = "DEFECTIVE"


        # ----------------------------------------------------
        # SVM PREDICTION CONFIDENCE
        # ----------------------------------------------------

        probabilities = svm_model.predict_proba(
            svm_features_scaled
        )[0]

        svm_confidence = np.max(
            probabilities
        ) * 100


        # ----------------------------------------------------
        # QUANTUM QSVC PREDICTION
        # ----------------------------------------------------

        qsvc_features_scaled = qsvc_scaler.transform(
            features_2d
        )

        qsvc_prediction = qsvc_model.predict(
            qsvc_features_scaled
        )[0]

        if qsvc_prediction == 0:
            qsvc_result = "GOOD"
        else:
            qsvc_result = "DEFECTIVE"


        # ====================================================
        # QUANTUM PCB INSPECTION
        # ====================================================

        st.divider()

        st.subheader("⚛️ Quantum PCB Inspection")

        if qsvc_result == "GOOD":

            st.success(
                "✅ PCB STATUS: GOOD"
            )

        else:

            st.error(
                "⚠️ PCB STATUS: DEFECTIVE"
            )

        st.metric(
            "Quantum QSVC Prediction",
            qsvc_result
        )


        # ====================================================
        # CLASSICAL SVM BASELINE
        # ====================================================

        st.divider()

        st.subheader("📊 Classical SVM Baseline")

        if svm_result == "GOOD":

            st.info(
                "SVM Prediction: GOOD"
            )

        else:

            st.warning(
                "SVM Prediction: DEFECTIVE"
            )

        st.metric(
            "SVM Prediction Confidence",
            f"{svm_confidence:.2f}%"
        )


        # ====================================================
        # EXTRACTED FEATURES
        # ====================================================

        st.divider()

        st.subheader("🔬 Extracted Feature Vector")

        st.write(
            "The uploaded PCB image is converted into "
            "a 10-dimensional numerical feature vector."
        )

        feature_data = {
            "Feature": FEATURE_NAMES,
            "Value": features
        }

        st.dataframe(
            feature_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()

        st.subheader(
            "📈 Model Performance"
        )

        st.write(
            "These metrics represent the performance of the "
            "classical SVM across the evaluation dataset."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{svm_metrics['accuracy'] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Precision",
                f"{svm_metrics['precision'] * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Recall",
                f"{svm_metrics['recall'] * 100:.2f}%"
            )

        with col4:

            st.metric(
                "F1 Score",
                f"{svm_metrics['f1'] * 100:.2f}%"
            )


        # ====================================================
        # QUANTUM INSPECTION ENGINE
        # ====================================================

        st.divider()

        st.subheader(
            "⚛️ Quantum Inspection Engine"
        )

        st.markdown(
            """
            ### Quantum ML Pipeline

            **PCB Image**
            → **Preprocessing**
            → **10-D Feature Vector**
            → **Quantum Feature Map**
            → **Quantum Kernel**
            → **QSVC**
            → **Quantum Prediction**
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.info(
                "🔢 10-D Feature Vector"
            )

            st.write(
                "Image converted into 10 numerical features."
            )

        with col2:

            st.info(
                "⚛️ Quantum Feature Map"
            )

            st.write(
                "Classical features are encoded into quantum states."
            )

        with col3:

            st.info(
                "🔗 Quantum Kernel"
            )

            st.write(
                "Quantum states are compared to calculate similarity."
            )

        st.success(
            "⚛️ Quantum QSVC Engine — Active"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "QuantumInspec | Quantum Machine Learning for PCB Defect Inspection"
)