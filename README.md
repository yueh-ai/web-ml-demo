# ML Browser Demo - Real-time Inference with ONNX Runtime

A lightweight demo showing how to run XGBoost models entirely in the browser using ONNX Runtime Web and React 16.

## Features

- 🚀 Real-time predictions running 100% client-side
- 🎚️ Interactive sliders for all features
- ⚡ Fast inference using WebAssembly
- 📦 Tiny model size (~47KB)
- 🔧 Built with React 16 and Material-UI

## Quick Start

### 1. Build the Model (if needed)
```bash
cd model-dev
uv sync
uv run python build_model.py
```

This creates:
- `web/public/model.onnx` - The ONNX model
- `web/public/model-package.json` - Model metadata

### 2. Run the Web App
```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## How It Works

1. **Python Side**: XGBoost model trained on California Housing dataset, exported to ONNX format
2. **Web Worker**: Runs ONNX Runtime in a separate thread for non-blocking inference
3. **React Hooks**: Clean abstraction for model loading and predictions
4. **Real-time UI**: Predictions update as you move the sliders

## Model Details

- **Algorithm**: XGBoost Regressor
- **Dataset**: California Housing (10k samples)
- **Features**: 8 inputs (income, house age, rooms, etc.)
- **Performance**: R² = 0.807
- **Size**: 47KB (uncompressed)
- **Training Time**: < 0.1 seconds on M1 Mac

## Architecture

```
model-dev/           # Python model training
├── build_model.py   # Trains and exports to ONNX
└── pyproject.toml   # UV dependencies

web/                 # React application  
├── public/
│   ├── model.onnx
│   └── model-package.json
└── src/
    ├── workers/     # Web Worker for inference
    ├── hooks/       # React hooks
    └── components/  # UI components
```

## Browser Compatibility

- Chrome 90+
- Firefox 89+
- Safari 15+
- Edge 90+

WebAssembly is required for ONNX Runtime.