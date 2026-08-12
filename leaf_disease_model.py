from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
HF_REPO_ID = "piyush2201/LeafDiseaseModel"
HF_MODEL_FILENAME = "leaf_disease_model.joblib"

MODEL_PATH = BASE_DIR / "static" / "models" / "leaf_disease_model.joblib"

MAX_IMAGES_PER_CLASS = 220
MODEL_VERSION = 2


def _summarize_channel(channel_values: np.ndarray) -> list[float]:
    channel_values = channel_values.astype(np.float32)
    mean = float(channel_values.mean())
    std = float(channel_values.std())
    skew = float(np.nan_to_num(np.mean(((channel_values - mean) / max(std, 1e-6)) ** 3)))
    kurtosis = float(np.nan_to_num(np.mean(((channel_values - mean) / max(std, 1e-6)) ** 4) - 3.0))
    q25 = float(np.percentile(channel_values, 25))
    q75 = float(np.percentile(channel_values, 75))
    return [mean, std, skew, kurtosis, q25, q75]


def _histogram_features(channel_values: np.ndarray, bins: int = 16) -> list[float]:
    histogram, _ = np.histogram(channel_values.ravel(), bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float32) / max(histogram.sum(), 1)
    return histogram.tolist()


def _quadrant_features(channel_values: np.ndarray) -> list[float]:
    height, width = channel_values.shape
    mid_h = max(height // 2, 1)
    mid_w = max(width // 2, 1)

    quadrants = [
        channel_values[:mid_h, :mid_w],
        channel_values[:mid_h, mid_w:],
        channel_values[mid_h:, :mid_w],
        channel_values[mid_h:, mid_w:],
    ]
    return [float(q.mean()) for q in quadrants] + [float(q.std()) for q in quadrants]


def extract_features(image_path: str | os.PathLike[str]) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB").resize((128, 128))
        rgb_array = np.array(rgb_image, dtype=np.float32) / 255.0
        hsv_array = np.array(rgb_image.convert("HSV"), dtype=np.float32) / 255.0

    if rgb_array.ndim != 3:
        rgb_array = np.stack([rgb_array] * 3, axis=-1)
    if hsv_array.ndim != 3:
        hsv_array = np.stack([hsv_array] * 3, axis=-1)

    feature_parts: list[np.ndarray] = []
    for array in (rgb_array, hsv_array):
        for channel in range(3):
            channel_values = array[:, :, channel]
            feature_parts.append(np.array(_summarize_channel(channel_values), dtype=np.float32))
            feature_parts.append(np.array(_histogram_features(channel_values), dtype=np.float32))
            feature_parts.append(np.array(_quadrant_features(channel_values), dtype=np.float32))

    gray = np.mean(rgb_array, axis=-1)
    grad_x = np.abs(np.gradient(gray, axis=1))
    grad_y = np.abs(np.gradient(gray, axis=0))
    gradient_features = np.array(
        [
            float(grad_x.mean()),
            float(grad_y.mean()),
            float(np.sqrt(grad_x**2 + grad_y**2).mean()),
            float(np.std(grad_x)),
            float(np.std(grad_y)),
        ],
        dtype=np.float32,
    )
    feature_parts.append(gradient_features)

    return np.concatenate(feature_parts).astype(np.float32)


def build_training_data() -> Tuple[np.ndarray, np.ndarray]:
    random.seed(42)
    features: list[np.ndarray] = []
    labels: list[str] = []

    class_dirs = sorted([path for path in TRAIN_DIR.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        image_paths = sorted(class_dir.glob("*"))
        selected_paths = image_paths[:MAX_IMAGES_PER_CLASS]
        if len(image_paths) > MAX_IMAGES_PER_CLASS:
            selected_paths = random.sample(image_paths, MAX_IMAGES_PER_CLASS)

        for image_path in selected_paths:
            try:
                feature_vector = extract_features(image_path)
            except Exception:
                continue
            features.append(feature_vector)
            labels.append(class_dir.name)

    if not features:
        raise RuntimeError("No training images were loaded from the dataset.")

    return np.stack(features), np.array(labels, dtype=object)


def train_model() -> Dict[str, object]:
    features, labels = build_training_data()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                ExtraTreesClassifier(
                 n_estimators=100,
                 random_state=42,
                  n_jobs=-1,
                 class_weight="balanced_subsample",
                 min_samples_leaf=1,
                 max_depth=30,
),
            ),
        ]
    )
    model.fit(features, encoded_labels)

    bundle = {"model": model, "label_encoder": label_encoder, "feature_version": MODEL_VERSION}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def ensure_model_exists() -> Dict[str, object]:
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
    )

    print("STARTING MODEL LOAD...", flush=True)

    bundle = joblib.load(model_path)

    print("MODEL LOADED SUCCESSFULLY!", flush=True)

    if bundle.get("feature_version") != MODEL_VERSION:
        raise RuntimeError(
            f"Model version mismatch. Expected {MODEL_VERSION}, "
            f"but found {bundle.get('feature_version')}."
        )

    return bundle


def predict_disease(image_path: str | os.PathLike[str]) -> Tuple[str, float]:
    bundle = ensure_model_exists()
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    feature_vector = extract_features(image_path).reshape(1, -1)
    encoded_prediction = model.predict(feature_vector)[0]
    predicted_label = label_encoder.inverse_transform([encoded_prediction])[0]
    probabilities = model.predict_proba(feature_vector)[0]
    prediction_index = int(encoded_prediction)
    confidence = float(probabilities[prediction_index])
    return predicted_label, confidence


def get_remedy_tip(label: str) -> str:
    label_lower = label.lower()

    if "healthy" in label_lower:
        return "The leaf looks healthy. Keep the plant well-watered, inspect it weekly, and avoid over-fertilizing."
    if "scab" in label_lower:
        return "Remove affected leaves, improve airflow, and apply a copper or sulfur-based fungicide early."
    if "black rot" in label_lower or "black_rot" in label_lower:
        return "Prune out infected growth, remove fallen fruit, and spray a fungicide during wet weather."
    if "rust" in label_lower:
        return "Remove infected parts, avoid overhead watering, and apply a rust-control fungicide."
    if "mildew" in label_lower:
        return "Improve airflow, reduce humidity around the plant, and use a mildew-specific fungicide."
    if "blight" in label_lower:
        return "Remove infected foliage immediately, water at the soil line, and use a labeled fungicide."
    if "bacterial" in label_lower:
        return "Sanitize pruning tools, remove infected leaves, and avoid wetting the foliage."
    if "spot" in label_lower:
        return "Prune the worst leaves, keep foliage dry, and use a disease-control spray if the spread continues."
    if "greening" in label_lower or "huanglongbing" in label_lower:
        return "Remove the affected tree if possible, avoid spreading the disease, and consult a local extension service."
    if "virus" in label_lower or "mosaic" in label_lower:
        return "Remove infected plants or leaves quickly and keep insects away to reduce spread."
    if "spider" in label_lower or "mites" in label_lower:
        return "Wash the leaves, increase humidity slightly, and use insecticidal soap if needed."
    return "Remove affected leaves, improve airflow, and contact a local agricultural extension service for a targeted treatment plan."
