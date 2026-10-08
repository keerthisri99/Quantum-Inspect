import streamlit as st
st.title("⚛️ Quantum-Inspect")

st.subheader(
    "Quantum-Powered PCB Defect Inspection"
)

st.write(
    "Upload a PCB image to analyze its "
    "defect status."
)
st.divider()

st.subheader("📷 Upload PCB Image")

uploaded_file = st.file_uploader(
    "Choose a PCB image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    st.image(
        image,
        caption="Uploaded PCB",
        use_container_width=True
    )

    if st.button("🔍 Inspect PCB"):

        # Extract the same 10 features
        features = extract_features(image)

        # Convert to 2D
        features_2d = features.reshape(1, -1)

        # Apply training scaler
        features_scaled = scaler.transform(
            features_2d
        )

        # Predict
        prediction = svm_model.predict(
            features_scaled
        )[0]

        if prediction == 0:
            result = "GOOD"
        else:
            result = "DEFECTIVE"

        probabilities = svm_model.predict_proba(
            features_scaled
        )[0]

        confidence = np.max(
            probabilities
        ) * 100

        st.divider()

        st.subheader(
            "🔍 Current PCB Inspection"
        )

        if result == "GOOD":
            st.success(
                "✅ PCB STATUS: GOOD"
            )
        else:
            st.error(
                "⚠️ PCB STATUS: DEFECTIVE"
            )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )
        st.divider()

        st.subheader(
            "🔬 Extracted Feature Vector"
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
st.divider()

st.subheader(
    "📊 Classical SVM Baseline"
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