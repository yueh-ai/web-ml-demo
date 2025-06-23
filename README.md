# ML Browser Demo - Real-time Inference with ONNX Runtime

A TypeScript-powered demo showing how to run XGBoost models entirely in the browser using ONNX Runtime Web, React, and Web Workers.

## Features

- 🚀 Real-time predictions running 100% client-side
- 🎚️ Interactive sliders for all features
- ⚡ Fast inference using WebAssembly
- 📦 Tiny model size (~47KB)
- 🔧 Built with React, TypeScript, and Material-UI
- 🔒 Type-safe message passing between main thread and worker
- 🎯 Strict TypeScript for better developer experience

## Quick Start

### 1. Build the Model (if needed)
```bash
cd model-dev
uv sync
uv run python build_model.py
```

This creates:
- `web/public/models/xgb_california_housing.onnx` - The ONNX model
- `web/public/models/xgb_california_housing_metadata.json` - Model metadata

### 2. Run the Web App
```bash
cd web
npm install
npm run dev
```

Open http://localhost:5173 in your browser (Vite default port).

### Build for Production
```bash
npm run build
npm run preview  # Test the production build
```

## How It Works

1. **Python Side**: XGBoost model trained on California Housing dataset, exported to ONNX format
2. **TypeScript Types**: Strongly typed interfaces for worker messages and data structures
3. **Web Worker**: Runs ONNX Runtime in a separate thread for non-blocking inference
4. **React Hooks**: Clean abstraction for model loading and predictions with full TypeScript support
5. **Real-time UI**: Predictions update as you move the sliders with type-safe event handling

## Model Details

- **Algorithm**: XGBoost Regressor
- **Dataset**: California Housing (10k samples)
- **Features**: 8 inputs (income, house age, rooms, etc.)
- **Performance**: R² = 0.807
- **Size**: 47KB (uncompressed)
- **Training Time**: < 0.1 seconds on M1 Mac

## Architecture

```
model-dev/                    # Python model training
├── build_model.py           # Trains and exports to ONNX
└── pyproject.toml           # UV dependencies

web/                         # React TypeScript application  
├── public/
│   └── models/
│       ├── xgb_california_housing.onnx
│       └── xgb_california_housing_metadata.json
├── src/
│   ├── types/              # TypeScript type definitions
│   │   ├── index.ts
│   │   ├── worker-imports.d.ts
│   │   └── worker-messages.ts
│   ├── workers/            # Web Worker for inference
│   │   └── predict.worker.ts
│   ├── hooks/              # React hooks with TypeScript
│   │   ├── useModelLoader.ts
│   │   └── usePredict.ts
│   └── components/         # UI components (TSX)
│       ├── App.tsx
│       ├── DemoPredictor.tsx
│       └── main.tsx
├── tsconfig.json           # TypeScript configuration
├── tsconfig.node.json      # TypeScript config for Node
└── vite.config.js          # Vite bundler configuration
```

## TypeScript Benefits

The project has been migrated to TypeScript for:
- **Type-safe worker communication**: Prevents runtime errors from message mismatches
- **Better IDE support**: Auto-completion and inline documentation
- **Compile-time error detection**: Catch issues before runtime
- **Self-documenting code**: Types serve as inline documentation

## Key Technologies

- **React 17**: UI framework with hooks
- **TypeScript 5.8**: Static typing for JavaScript
- **Vite**: Fast build tool with HMR
- **ONNX Runtime Web**: Machine learning inference in the browser
- **Material-UI**: React component library
- **Web Workers**: Background thread for non-blocking inference

## Browser Compatibility

- Chrome 90+
- Firefox 89+
- Safari 15+
- Edge 90+

WebAssembly and Web Workers are required for ONNX Runtime.