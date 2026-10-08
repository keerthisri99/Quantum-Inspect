import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


FEATURE_NAMES = [
    "Intensity Minimum",
    "Intensity Maximum",
    "Intensity Mean",
    "Intensity Std",
    "Edge Count",
    "Edge Density",
    "Texture Contrast",
    "Entropy",
    "GLCM Homogeneity",
    "GLCM Contrast"
]


def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (256, 256))

    # 1–4. Intensity features
    minimum = np.min(gray)
    maximum = np.max(gray)
    mean = np.mean(gray)
    standard_deviation = np.std(gray)

    # 5. Edge count
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)

    # 6. Edge density
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = edge_count / total_pixels

    # 7. Texture contrast
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_contrast = np.std(laplacian)

    # 8. Entropy
    histogram = cv2.calcHist(
        [gray], [0], None, [256], [0, 256]
    )

    probabilities = histogram / np.sum(histogram)
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    # 9–10. GLCM features
    glcm_image = (gray // 8).astype(np.uint8)

    glcm = graycomatrix(
        glcm_image,
        distances=[1],
        angles=[0],
        levels=32,
        symmetric=True,
        normed=True
    )

    glcm_homogeneity = graycoprops(
        glcm, "homogeneity"
    )[0, 0]

    glcm_contrast = graycoprops(
        glcm, "contrast"
    )[0, 0]

    features = np.array([
        minimum,
        maximum,
        mean,
        standard_deviation,
        edge_count,
        edge_density,
        texture_contrast,
        entropy,
        glcm_homogeneity,
        glcm_contrast
    ], dtype=np.float32)

    return features