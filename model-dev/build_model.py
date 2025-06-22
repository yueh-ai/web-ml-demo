#!/usr/bin/env python3
"""
Build lightweight XGBoost model for browser demo
Optimized for fast training on M1 Mac
"""
import json
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "web" / "public"
OUT.mkdir(parents=True, exist_ok=True)

print("🚀 Starting model build...")
start_time = time.time()

# 1. Load California Housing dataset (faster than Boston)
print("📊 Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# Use only 10k samples for faster training on M1
sample_size = min(10000, len(X))
indices = np.random.choice(len(X), sample_size, replace=False)
X = X[indices]
y = y[indices]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")

# 2. Train lightweight XGBoost model
print("🔧 Training XGBoost model (optimized for M1)...")
model = xgb.XGBRegressor(
    n_estimators=50,      # Reduced from 100-200
    max_depth=4,          # Keep shallow for smaller model
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,            # Use all M1 cores
    tree_method='hist'    # Faster on M1
)

train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start
print(f"✅ Training completed in {train_time:.2f} seconds")

# Quick evaluation
score = model.score(X_test, y_test)
print(f"R² score on test set: {score:.3f}")

# 3. Convert to ONNX
print("🔄 Converting to ONNX format...")
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_xgboost(
    model, 
    initial_types=initial_type,
    target_opset=15  # Compatible with current ONNX version
)

# Save model
onnx_path = OUT / "model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# 4. Try quantization, but fall back to regular model if it fails
print("📦 Attempting model quantization...")
quantized_path = OUT / "model.int8.onnx"
try:
    quantize_dynamic(
        str(onnx_path),
        str(quantized_path),
        weight_type=QuantType.QUInt8,
        optimize_model=False  # Skip optimization to avoid issues
    )
    # If successful, remove original
    onnx_path.unlink()
    final_model_path = quantized_path
except Exception as e:
    print(f"⚠️  Quantization failed: {e}")
    print("Using unquantized model instead")
    final_model_path = onnx_path
    quantized_path = onnx_path

# Check file sizes
original_size = onnx_path.stat().st_size / 1024
quantized_size = quantized_path.stat().st_size / 1024
print(f"Original model size: {original_size:.1f} KB")
print(f"Quantized model size: {quantized_size:.1f} KB")
print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

# 5. Generate metadata for frontend
feature_names = [
    "MedInc",        # Median income
    "HouseAge",      # House age
    "AveRooms",      # Average rooms
    "AveBedrms",     # Average bedrooms
    "Population",    # Population
    "AveOccup",      # Average occupancy
    "Latitude",      # Latitude
    "Longitude"      # Longitude
]

# Define reasonable ranges for sliders
feature_ranges = {
    "MedInc": {"min": 0.5, "max": 15.0, "step": 0.5, "default": 3.0},
    "HouseAge": {"min": 1, "max": 52, "step": 1, "default": 25},
    "AveRooms": {"min": 1, "max": 10, "step": 0.5, "default": 5},
    "AveBedrms": {"min": 0.5, "max": 5, "step": 0.1, "default": 1.0},
    "Population": {"min": 100, "max": 10000, "step": 100, "default": 3000},
    "AveOccup": {"min": 1, "max": 6, "step": 0.1, "default": 3.0},
    "Latitude": {"min": 32, "max": 42, "step": 0.1, "default": 34.0},
    "Longitude": {"min": -125, "max": -114, "step": 0.1, "default": -118.0}
}

metadata = {
    "model": final_model_path.name,
    "version": "1.0.0",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "input_names": ["float_input"],
    "output_name": "variable",
    "feature_order": feature_names,
    "feature_ranges": feature_ranges,
    "model_info": {
        "type": "XGBoost Regressor",
        "task": "California House Price Prediction",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "training_samples": len(X_train),
        "r2_score": round(score, 3)
    }
}

metadata_path = OUT / "model-package.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

total_time = time.time() - start_time
model_size = final_model_path.stat().st_size / 1024
print(f"\n✅ Model build completed in {total_time:.2f} seconds!")
print(f"📁 Output files saved to: {OUT}")
print(f"   - {final_model_path.name} ({model_size:.1f} KB)")
print(f"   - {metadata_path.name}")