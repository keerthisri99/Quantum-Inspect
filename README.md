Quantum-Inspect ⚛️

QuantumInspect is an experimental quantum machine-learning project exploring PCB (Printed Circuit Board) defect classification using image-derived features. It compares a classical Support Vector Machine (SVM) baseline with a Quantum Support Vector Classifier (QSVC) using a fidelity quantum kernel.

# Project Overview
PCB inspection is important in electronics manufacturing. This project investigates whether a quantum-kernel-based classifier can distinguish between PCB images labelled GOOD and DEFECTIVE.

# What We Explore
- Classical image feature extraction
- Classical SVM baseline
- Qiskit "ZZFeatureMap"
- "FidelityQuantumKernel" and QSVC
- Mutual-information feature selection
- Accuracy, precision, recall, F1 score, and confusion matrix
- A Streamlit interface for testing uploaded PCB images

# How It Works
1. Input: Load a PCB image.
2. Feature extraction: Calculate image statistics, edge information, entropy, contrast, and GLCM texture features.
3. Classical baseline: Train and evaluate an SVM.
4. Quantum ML experiment: Encode selected features with a ZZ feature map and use a fidelity quantum kernel with QSVC.
5. Evaluation: Compare predictions using classification metrics.
6. Demo: Upload a PCB image through the Streamlit interface.

Image processing and feature extraction are classical. The quantum-ML component is the quantum feature-map/kernel approach used by QSVC. Do not describe the code as running on physical quantum hardware unless a hardware backend has been explicitly configured and verified.

# Technology Stack
- Python
- NumPy
- scikit-learn
- Qiskit
- Qiskit Machine Learning
- OpenCV
- scikit-image
- Joblib
- Streamlit

# Install dependencies
Install the packages required by your code

A starting point is:

pip install numpy scikit-learn joblib streamlit qiskit qiskit-machine-learning opencv-python scikit-image

Package compatibility depends on Python and Qiskit versions. Use a tested environment and consider saving exact versions in "requirements.txt".

# Prepare the dataset
The training script expects:

data/X.npy
data/y.npy

- "X.npy" contains the numerical feature matrix.
- "y.npy" contains the corresponding labels.

Important: Labels must reflect the dataset's ground truth. Do not assign labels based on file order or alternate them artificially, because that makes evaluation metrics unreliable. Follow the dataset's terms when sharing or redistributing it.

# Train the quantum model

Run from the project root:

python Notebooks/quantum_model.py

The training script searches feature subsets and QSVC parameters using cross-validation, evaluates the selected configuration on a held-out test split, and saves model artifacts under "models/".

If the script loads existing artifacts instead of retraining, configure its main execution block to call "train_model()" directly for an intentional retraining run. Retraining overwrites QSVC artifacts, so back them up first if you need the previous versions.

# Run the Streamlit app

python -m streamlit run app.py

Open the local URL printed in the terminal, usually "http://localhost:8501".

If the app needs saved model artifacts, train the models first and verify that the expected files exist in "models/".

Evaluation Metrics
- Accuracy: Fraction of predictions that are correct.
- Precision: Fraction of predicted positives that are truly positive.
- Recall: Fraction of actual positives correctly identified.
- F1 score: Harmonic mean of precision and recall.
- Confusion matrix: Counts of correct and incorrect predictions by class.

For defect inspection, recall for the defective class is especially important because missed defects can be costly. State which class is treated as the positive class when interpreting precision and recall.

For a fair comparison, use valid ground-truth labels, consistent data splits, and comparable evaluation procedures for SVM and QSVC. Do not compare one model's cross-validation accuracy with the other's held-out test accuracy.

# Limitations
- This is an experimental quantum-ML application, not proof of quantum advantage.
- A quantum feature map does not guarantee better accuracy.
- Small datasets or unreliable labels can produce misleading metrics.
- Simulator results are not the same as results from physical quantum hardware.
- Performance and runtime depend on the data, feature selection, hyperparameters, software versions, and execution backend.

# Future Scope
- Verify and document dataset ground-truth labels.
- Evaluate both models with a consistent, reproducible protocol.
- Repeat experiments across multiple splits or random seeds.
- Record software versions, runtime, and results.
- Explore additional feature maps and classical baselines.
- Investigate hardware execution if suitable access is available.

# Team and Acknowledgements
Built as part of our participation in Qiskit Fall Fest Round 2.
