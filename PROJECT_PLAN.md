# Interactive ML Browser Demo - Production Blueprint

## Overview

A production-grade web application for running scikit-learn and XGBoost models entirely in the browser using React 16+ with ONNX Runtime Web. Features real-time predictions with interactive parameter adjustment and WebGPU acceleration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Model Development (Python)                                  │
├─────────────────────────────────────────────────────────────┤
│ • Train models (scikit-learn/XGBoost)                      │
│ • Convert to ONNX format                                   │
│ • Quantize for performance (optional)                      │
│ • Generate metadata package                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ CI/CD Pipeline                                              │
├─────────────────────────────────────────────────────────────┤
│ • Run model accuracy tests                                 │
│ • Build React application                                  │
│ • Bundle ONNX models + WASM runtime                        │
│ • Deploy to CDN/S3                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Browser Runtime                                             │
├─────────────────────────────────────────────────────────────┤
│ • React 16.8+ with Hooks                                   │
│ • ONNX Runtime Web (WebGPU/WASM)                          │
│ • Web Workers for non-blocking inference                   │
│ • Real-time UI updates                                     │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Python Side

- **scikit-learn** / **XGBoost** for model training
- **skl2onnx** / **onnxmltools** for ONNX conversion
- **onnxruntime** for quantization and validation
- **UV** for dependency management

### Frontend

- **React 16.14+** (Hooks required)
- **ONNX Runtime Web** for browser inference
- **Web Workers** for non-blocking computation
- **Material-UI** for slider components
- **Vite** for bundling and development

## Project Structure

```
web-ml-demo/
├── model-dev/                    # Python model development
│   ├── pyproject.toml           # UV dependencies
│   ├── build_model.py           # Model training & ONNX export
│   ├── test_model_accuracy.py   # Accuracy parity tests
│   └── models/                  # Model artifacts
├── web/                         # React application
│   ├── package.json
│   ├── vite.config.js
│   ├── public/
│   │   ├── model.onnx          # Quantized model
│   │   └── model-package.json  # Model metadata
│   └── src/
│       ├── workers/
│       │   └── predict.worker.js
│       ├── hooks/
│       │   ├── useModelLoader.js
│       │   └── usePredict.js
│       ├── components/
│       │   ├── DemoPredictor.jsx
│       │   └── VisualizationPanel.jsx
│       └── App.jsx
├── .github/
│   └── workflows/
│       └── web.yml             # CI/CD pipeline
└── README.md
```

## Implementation Phases

### Phase 1: Model Development Pipeline

1. Create `build_model.py` for training and ONNX conversion
2. Implement quantization for model size reduction
3. Generate `model-package.json` with metadata
4. Add accuracy parity tests

### Phase 2: Frontend Infrastructure

1. Set up React 16 with Vite
2. Configure ONNX Runtime Web with WebGPU support
3. Implement Web Worker for inference
4. Create custom hooks for model loading and prediction

### Phase 3: Interactive UI

1. Build parameter control panel with sliders
2. Add real-time visualization components
3. Implement debouncing for smooth updates
4. Add loading states and error handling

### Phase 4: CI/CD & Deployment

1. GitHub Actions workflow for automated builds
2. Model versioning and A/B testing support
3. CDN deployment with cache optimization
4. Performance monitoring integration

## Key Features

### Model Agnostic Design

- Swap models by updating `model-package.json`
- Support for multiple models per page
- Dynamic feature schema from metadata

### Performance Optimizations

- WebGPU acceleration (with WASM fallback)
- Dynamic quantization (int8) for smaller models
- Web Workers prevent UI blocking
- 30ms debounce for smooth interactions

### Production Considerations

- Bundle size < 1MB for models (gzipped)
- Accuracy parity testing in CI
- Device-aware performance tuning
- Telemetry for prediction latency

## Commands

```bash
# Model Development
cd model-dev
uv sync
uv run python build_model.py

# Frontend Development
cd web
npm install
npm run dev

# Production Build
npm run build

# Run Tests
uv run pytest test_model_accuracy.py
npm test
```

## Extensibility Matrix

| Change Scenario   | Implementation Approach            |
| ----------------- | ---------------------------------- |
| New Algorithm     | Convert to ONNX → Update metadata  |
| Multiple Models   | Spawn multiple workers             |
| Feature Evolution | Update `feature_order` in metadata |
| A/B Testing       | Version models, use feature flags  |
| Low-end Devices   | Serve both int8 and fp32 variants  |

## Security & Privacy

- Models run entirely client-side (no server calls)
- Model parameters are visible to users
- Consider server-side inference for proprietary models

## Success Metrics

- Initial load time < 3 seconds
- Prediction latency < 50ms
- 60fps UI interactions
- Model size < 1MB compressed
