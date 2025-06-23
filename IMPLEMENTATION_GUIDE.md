# Browser-Based Machine Learning Implementation Guide

This guide explains how to integrate client-side machine learning capabilities into your web application using ONNX Runtime Web. The implementation demonstrates running ML models directly in the browser, eliminating the need for server-side inference.

## Project Overview

This project provides a complete example of browser-based machine learning using:
- **ONNX Runtime Web** for model inference
- **TypeScript** for type safety
- **React** with Material-UI for the UI
- **Web Workers** for non-blocking inference
- **Vite** for fast development and building

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Model Preparation](#model-preparation)
5. [Core Implementation](#core-implementation)
6. [Running the Demo](#running-the-demo)
7. [Framework Integration](#framework-integration)
8. [Build Tool Configuration](#build-tool-configuration)
9. [Testing & CI/CD](#testing--cicd)
10. [Performance & Optimization](#performance--optimization)
11. [Production Considerations](#production-considerations)

## Core Concepts

### What You're Building

A feature that runs machine learning predictions entirely in the user's browser:

- **No API calls** for predictions
- **Zero latency** inference
- **Complete privacy** - data never leaves the device
- **Offline capable** once model is loaded
- **Infinitely scalable** - no server costs

### Key Technologies

- **ONNX**: Open standard for ML model representation
- **ONNX Runtime Web**: JavaScript library for running ONNX models
- **Web Workers**: Background threads for non-blocking inference
- **WebAssembly**: High-performance execution environment

## Architecture Overview

The implementation consists of three main components:

```
┌─────────────────────────────────────────────────────────┐
│                    React Application                     │
│                                                          │
│  ┌──────────────┐     ┌─────────────┐     ┌─────────┐  │
│  │ UI Component │ <-> │ React Hooks │ <-> │ Worker  │  │
│  │ (Material-UI)│     │ (TypeScript)│     │ Thread  │  │
│  └──────────────┘     └─────────────┘     └─────────┘  │
│                                                  │        │
│                                                  ▼        │
│                                            ┌─────────┐   │
│                                            │ ONNX    │   │
│                                            │ Runtime │   │
│                                            └─────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
web-ml-demo/
├── model-dev/               # Python model development
│   ├── build_model.py      # Train and export XGBoost model
│   └── requirements.txt    # Python dependencies
│
└── web/                    # React TypeScript application
    ├── public/
    │   ├── model.onnx             # ONNX model file (48KB)
    │   └── model-package.json     # Model metadata
    ├── src/
    │   ├── components/
    │   │   └── DemoPredictor.tsx  # Main prediction UI
    │   ├── hooks/
    │   │   ├── useModelLoader.ts  # Model loading hook
    │   │   └── usePredict.ts      # Prediction hook
    │   ├── types/
    │   │   ├── index.ts           # Type exports
    │   │   ├── features.ts        # Feature types
    │   │   ├── worker-messages.ts # Worker communication types
    │   │   └── worker-imports.d.ts# Worker import declarations
    │   ├── workers/
    │   │   └── predict.worker.ts  # Web Worker for inference
    │   └── App.tsx                # Main app component
    ├── package.json               # Dependencies
    ├── tsconfig.json             # TypeScript config
    └── vite.config.js            # Vite bundler config
```

## Model Preparation

### Current Implementation: XGBoost Model

The project includes a complete example using XGBoost for California house price prediction:

```python
# model-dev/build_model.py
import json
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# Load and prepare data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train lightweight XGBoost model
model = XGBRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Export to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = to_onnx(model, initial_types=initial_type, target_opset=12)

# Save model
with open('../web/public/model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Generate metadata
metadata = {
    "model": "model.onnx",
    "version": "1.0.0",
    "input_names": ["float_input"],
    "output_name": "variable",
    "feature_order": data.feature_names,
    "feature_ranges": {
        # Feature-specific min/max/default values
        "MedInc": {"min": 0.5, "max": 15.0, "step": 0.5, "default": 3.0},
        "HouseAge": {"min": 1, "max": 52, "step": 1, "default": 25},
        # ... etc
    },
    "model_info": {
        "type": "XGBoost Regressor",
        "task": "California House Price Prediction",
        "training_samples": len(X_train),
        "r2_score": model.score(X_test, y_test)
    }
}

with open('../web/public/model-package.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Adapting for Your Own Model

1. **Train your model** using your preferred framework
2. **Export to ONNX** using the appropriate converter:
   - scikit-learn/XGBoost: `skl2onnx`
   - TensorFlow/Keras: `tf2onnx`
   - PyTorch: `torch.onnx.export`
3. **Create metadata** describing inputs, outputs, and feature ranges
4. **Place files** in `web/public/` directory

## Core Implementation

### Step 1: Install Dependencies

```bash
cd web
npm install
```

Current dependencies:
- `onnxruntime-web` - ONNX Runtime for browser
- `react` & `react-dom` - React framework
- `@mui/material` - Material-UI components
- `typescript` - Type safety
- `vite` - Build tool

### Step 2: TypeScript Types

The project uses comprehensive TypeScript types for safety:

```typescript
// types/worker-messages.ts
export interface ModelMetadata {
  model: string;
  version: string;
  created: string;
  input_names: string[];
  output_name: string;
  feature_order: string[];
  feature_ranges: Record<string, FeatureRange>;
  model_info: {
    type: string;
    task: string;
    n_estimators: number;
    max_depth: number;
    training_samples: number;
    r2_score: number;
  };
}

export interface FeatureRange {
  min: number;
  max: number;
  step: number;
  default: number;
}

export type FeatureData = Record<string, number>;

// Worker communication types
export interface LoadModelMessage {
  event: 'load';
  meta: ModelMetadata;
}

export interface PredictMessage {
  event: 'predict';
  features: FeatureData;
}

export interface LoadedMessage {
  event: 'loaded';
  success: boolean;
  error?: string;
}

export interface PredictionMessage {
  event: 'prediction';
  success: boolean;
  value?: number;
  error?: string;
}
```

### Step 3: Web Worker Implementation

The worker handles model loading and inference in a background thread:

```typescript
// workers/predict.worker.ts
import * as ort from 'onnxruntime-web';
import type { 
  WorkerIncomingMessage, 
  WorkerOutgoingMessage, 
  ModelMetadata 
} from '../types';

let session: ort.InferenceSession | null = null;
let meta: ModelMetadata | null = null;

self.onmessage = async ({ data }: MessageEvent<WorkerIncomingMessage>) => {
  switch (data.event) {
    case 'load':
      try {
        meta = data.meta;
        
        // Create inference session
        session = await ort.InferenceSession.create(`/${meta.model}`, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        
        const response: WorkerOutgoingMessage = { 
          event: 'loaded', 
          success: true 
        };
        postMessage(response);
      } catch (error) {
        const errorResponse: WorkerOutgoingMessage = { 
          event: 'loaded', 
          success: false, 
          error: error instanceof Error ? error.message : 'Unknown error' 
        };
        postMessage(errorResponse);
      }
      break;

    case 'predict':
      if (!session || !meta) {
        postMessage({ 
          event: 'prediction', 
          success: false, 
          error: 'Model not loaded' 
        });
        return;
      }

      try {
        // Extract features in correct order
        const features = meta.feature_order.map(name => data.features[name]);
        
        // Create tensor
        const inputTensor = new ort.Tensor(
          'float32',
          Float32Array.from(features),
          [1, features.length]
        );
        
        // Run inference
        const feeds: Record<string, ort.Tensor> = { 
          [meta.input_names[0]]: inputTensor 
        };
        const results = await session.run(feeds);
        
        // Extract prediction
        const prediction = results[meta.output_name].data[0] as number;
        
        postMessage({ 
          event: 'prediction', 
          success: true, 
          value: prediction 
        });
      } catch (error) {
        postMessage({ 
          event: 'prediction', 
          success: false, 
          error: error instanceof Error ? error.message : 'Unknown error' 
        });
      }
      break;
  }
};
```

### Step 4: React Hooks

The project uses custom hooks for clean model management:

```typescript
// hooks/useModelLoader.ts
import { useEffect, useState, useRef } from 'react';
import type { ModelMetadata, WorkerOutgoingMessage } from '../types';

export function useModelLoader() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    async function initModel() {
      try {
        // Fetch metadata
        const response = await fetch('/model-package.json');
        const metadata = await response.json() as ModelMetadata;
        setMeta(metadata);
        
        // Import worker using Vite's ?worker syntax
        const PredictWorker = await import('../workers/predict.worker.ts?worker');
        const worker = new PredictWorker.default();
        workerRef.current = worker;
        
        // Handle messages
        worker.addEventListener('message', ({ data }: MessageEvent<WorkerOutgoingMessage>) => {
          if (data.event === 'loaded') {
            if (data.success) {
              setReady(true);
              setLoading(false);
            } else {
              setError(data.error || 'Failed to load model');
              setLoading(false);
            }
          }
        });
        
        // Load model
        worker.postMessage({ event: 'load', meta: metadata });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    }
    
    initModel();
    
    return () => {
      workerRef.current?.terminate();
    };
  }, []);
  
  return { meta, ready, loading, error, worker: workerRef.current };
}

// hooks/usePredict.ts
import { useEffect, useState, useRef } from 'react';
import type { ModelMetadata, FeatureData, WorkerOutgoingMessage } from '../types';

export function usePredict({ worker, meta, features }: {
  worker: Worker | null;
  meta: ModelMetadata | null;
  features: FeatureData | null;
}) {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!worker || !meta || !features) return;

    // Debounce predictions
    if (debounceRef.current) clearTimeout(debounceRef.current);
    
    debounceRef.current = setTimeout(() => {
      setPredicting(true);
      setError(null);
      worker.postMessage({ event: 'predict', features });
    }, 50);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [features, worker, meta]);

  useEffect(() => {
    if (!worker) return;

    const handleMessage = ({ data }: MessageEvent<WorkerOutgoingMessage>) => {
      if (data.event === 'prediction') {
        if (data.success) {
          setPrediction(data.value ?? null);
          setPredicting(false);
        } else {
          setError(data.error || 'Prediction failed');
          setPredicting(false);
        }
      }
    };

    worker.addEventListener('message', handleMessage);
    return () => worker.removeEventListener('message', handleMessage);
  }, [worker]);

  return { prediction, predicting, error };
}
```

### Step 5: React Component

The main UI component uses Material-UI for a polished interface:

```typescript
// components/DemoPredictor.tsx
import { useState, useEffect } from 'react';
import {
  Container, Paper, Typography, Slider,
  Grid, Box, CircularProgress, Alert
} from '@mui/material';
import { useModelLoader } from '../hooks/useModelLoader';
import { usePredict } from '../hooks/usePredict';
import type { FeatureData } from '../types';

export default function DemoPredictor() {
  const { meta, ready, loading, error: loadError, worker } = useModelLoader();
  const [features, setFeatures] = useState<FeatureData | null>(null);

  // Initialize features
  useEffect(() => {
    if (meta?.feature_ranges) {
      const initialFeatures: FeatureData = {};
      Object.entries(meta.feature_ranges).forEach(([name, range]) => {
        initialFeatures[name] = range.default;
      });
      setFeatures(initialFeatures);
    }
  }, [meta]);

  const { prediction, predicting, error: predictError } = usePredict({
    worker, meta, features
  });

  const handleSliderChange = (featureName: string) => 
    (_: Event, value: number | number[]) => {
      setFeatures(prev => ({
        ...prev!,
        [featureName]: value as number
      }));
    };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ padding: 4, marginTop: 4 }}>
        <Paper sx={{ padding: 4 }}>
          <Box display="flex" justifyContent="center" minHeight={300}>
            <CircularProgress size={60} />
          </Box>
        </Paper>
      </Container>
    );
  }

  if (loadError) {
    return (
      <Container maxWidth="md" sx={{ padding: 4, marginTop: 4 }}>
        <Paper sx={{ padding: 4 }}>
          <Alert severity="error">Failed to load model: {loadError}</Alert>
        </Paper>
      </Container>
    );
  }

  if (!ready || !features || !meta) return null;

  return (
    <Container maxWidth="md" sx={{ padding: 4, marginTop: 4 }}>
      <Paper sx={{ padding: 4 }}>
        <Typography variant="h4" gutterBottom>
          California House Price Prediction
        </Typography>
        
        <Typography variant="body1" color="textSecondary" paragraph>
          Adjust the sliders below to see real-time predictions using an XGBoost 
          model running entirely in your browser.
        </Typography>

        <Grid container spacing={3}>
          {Object.entries(meta.feature_ranges).map(([name, range]) => (
            <Grid item xs={12} sm={6} key={name}>
              <Box mb={3}>
                <Typography gutterBottom>
                  {name}: <strong>
                    {features[name].toFixed(range.step < 1 ? 1 : 0)}
                  </strong>
                </Typography>
                <Slider
                  value={features[name]}
                  min={range.min}
                  max={range.max}
                  step={range.step}
                  onChange={handleSliderChange(name)}
                  valueLabelDisplay="auto"
                />
              </Box>
            </Grid>
          ))}
        </Grid>

        <Box mt={4} p={3} bgcolor="primary.main" color="primary.contrastText" 
             borderRadius={1} textAlign="center">
          <Typography variant="h6">Predicted House Price</Typography>
          <Typography variant="h3">
            {predicting ? (
              <CircularProgress size={40} color="inherit" />
            ) : prediction !== null ? (
              `$${(prediction * 100000).toLocaleString()}`
            ) : '—'}
          </Typography>
          <Typography variant="caption">
            (Price in hundreds of thousands of dollars)
          </Typography>
        </Box>

        <Box mt={3}>
          <Typography variant="caption" color="textSecondary">
            Model: {meta.model_info.type} | 
            R² Score: {meta.model_info.r2_score.toFixed(3)} | 
            Training Samples: {meta.model_info.training_samples.toLocaleString()}
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
}
```

## Running the Demo

### Step 1: Build the Model

```bash
cd model-dev
pip install -r requirements.txt
python build_model.py
```

### Step 2: Run the Web App

```bash
cd web
npm install
npm run dev
```

Visit http://localhost:3001 to see the demo in action.

## Framework Integration

### React Integration

```javascript
// hooks/useMLModel.js
import { useState, useEffect, useRef } from "react";
import { MLService } from "../services/ml-service";

export function useMLModel(modelUrl, metadataUrl) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const mlServiceRef = useRef(null);

  useEffect(() => {
    const mlService = new MLService();
    mlServiceRef.current = mlService;

    mlService
      .initialize(modelUrl, metadataUrl)
      .then((result) => {
        setMetadata(result.metadata);
        setIsLoading(false);
      })
      .catch((err) => {
        setError(err);
        setIsLoading(false);
      });

    return () => {
      mlService.dispose();
    };
  }, [modelUrl, metadataUrl]);

  const predict = async (inputData) => {
    if (!mlServiceRef.current) {
      throw new Error("Model not loaded");
    }
    return mlServiceRef.current.predict(inputData);
  };

  return { isLoading, error, metadata, predict };
}

// Component usage
function MLPredictor() {
  const { isLoading, error, metadata, predict } = useMLModel(
    "/models/model.onnx",
    "/models/model-metadata.json"
  );

  // Your UI logic here
}
```

### Vue Integration

```javascript
// composables/useMLModel.js
import { ref, onMounted, onUnmounted } from "vue";
import { MLService } from "../services/ml-service";

export function useMLModel(modelUrl, metadataUrl) {
  const isLoading = ref(true);
  const error = ref(null);
  const metadata = ref(null);
  let mlService = null;

  onMounted(async () => {
    mlService = new MLService();

    try {
      const result = await mlService.initialize(modelUrl, metadataUrl);
      metadata.value = result.metadata;
      isLoading.value = false;
    } catch (err) {
      error.value = err;
      isLoading.value = false;
    }
  });

  onUnmounted(() => {
    if (mlService) {
      mlService.dispose();
    }
  });

  const predict = async (inputData) => {
    if (!mlService) {
      throw new Error("Model not loaded");
    }
    return mlService.predict(inputData);
  };

  return { isLoading, error, metadata, predict };
}
```

### Angular Integration

```typescript
// services/ml.service.ts
import { Injectable } from "@angular/core";
import { BehaviorSubject, Observable } from "rxjs";
import { MLService } from "./ml-service";

@Injectable({
  providedIn: "root",
})
export class MLModelService {
  private mlService: MLService;
  private loadingSubject = new BehaviorSubject<boolean>(true);
  private metadataSubject = new BehaviorSubject<any>(null);

  loading$ = this.loadingSubject.asObservable();
  metadata$ = this.metadataSubject.asObservable();

  async initialize(modelUrl: string, metadataUrl: string): Promise<void> {
    this.mlService = new MLService();

    try {
      const result = await this.mlService.initialize(modelUrl, metadataUrl);
      this.metadataSubject.next(result.metadata);
      this.loadingSubject.next(false);
    } catch (error) {
      this.loadingSubject.next(false);
      throw error;
    }
  }

  async predict(inputData: number[]): Promise<number[]> {
    if (!this.mlService) {
      throw new Error("Model not loaded");
    }
    return this.mlService.predict(inputData);
  }

  ngOnDestroy(): void {
    if (this.mlService) {
      this.mlService.dispose();
    }
  }
}
```

## Build Tool Configuration

### Vite Configuration

The project uses Vite for fast development and optimized builds:

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  assetsInclude: ['**/*.onnx'],
  worker: {
    format: 'es'
  }
})
```

### TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable", "WebWorker"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### Worker Import Declaration

```typescript
// types/worker-imports.d.ts
declare module '*?worker' {
  const workerConstructor: {
    new (): Worker;
  };
  export default workerConstructor;
}
```

## Key Implementation Considerations

### Worker Communication Patterns

When implementing ML in the browser, proper worker communication is crucial:

1. **Type-Safe Messages**: Use TypeScript interfaces for all worker messages
2. **Error Handling**: Always handle both success and error cases
3. **State Management**: Track loading, ready, and error states
4. **Memory Management**: Terminate workers when components unmount

### Performance Optimization

1. **Model Size**: Keep models under 50MB for fast loading
2. **Debouncing**: Debounce predictions to avoid overwhelming the worker
3. **Web Workers**: Always run inference in workers to keep UI responsive
4. **WASM Execution**: Use WebAssembly backend for optimal performance

### Browser Compatibility

- **Modern Browsers**: Chrome, Edge, Firefox, Safari (latest versions)
- **Required Features**: WebAssembly, Web Workers, ES Modules
- **Fallbacks**: Consider server-side inference for unsupported browsers

### Security Considerations

1. **Model Protection**: Models are downloaded to client - consider if this is acceptable
2. **Input Validation**: Always validate inputs before inference
3. **CORS Headers**: Ensure proper CORS configuration for model files
4. **Content Security Policy**: May need adjustments for Web Workers

## Production Deployment

### Building for Production

```bash
cd web
npm run build
```

The build output will be in the `dist` directory with:
- Optimized JavaScript bundles
- ONNX model file
- Static assets

### Serving Considerations

1. **CORS Headers**: Ensure model files are served with appropriate CORS headers
2. **CDN**: Consider using a CDN for model files to reduce latency
3. **Compression**: Enable gzip/brotli for model metadata (not ONNX files)
4. **Caching**: Set appropriate cache headers for model files

### Monitoring

Track key metrics in production:
- Model load time
- Inference time per prediction
- Error rates
- Browser/device compatibility

## Adapting for Your Use Case

### 1. Different Model Types

The architecture supports any ONNX-compatible model:
- **Classification**: Return class probabilities instead of single value
- **Object Detection**: Handle image inputs and bounding box outputs
- **NLP Models**: Process text inputs with tokenization

### 2. Different Frameworks

While this demo uses React, the core concepts work with:
- **Vue.js**: Use composables instead of hooks
- **Angular**: Implement as services with observables
- **Vanilla JS**: Use the worker directly without framework

### 3. Advanced Features

Consider adding:
- **Model versioning**: Support multiple model versions
- **A/B testing**: Compare different models in production
- **Offline support**: Cache models with Service Workers
- **Model updates**: Hot-reload models without page refresh

## Troubleshooting

### Common Issues

1. **"SharedArrayBuffer not defined"**
   - Add CORS headers: `Cross-Origin-Embedder-Policy: require-corp`
   - Use `ort.env.wasm.numThreads = 1` to disable threading

2. **"Model file not found"**
   - Check public directory structure
   - Verify build process includes ONNX files
   - Check network tab for 404 errors

3. **"Slow inference"**
   - Reduce model complexity
   - Enable SIMD: `ort.env.wasm.simd = true`
   - Consider quantization

4. **"Memory issues"**
   - Dispose tensors after use
   - Terminate workers when done
   - Monitor memory usage

## Summary

This implementation demonstrates a production-ready approach to running ML models in the browser:

### Key Achievements
- ✅ **Zero-latency inference** - No network calls needed
- ✅ **Privacy-preserving** - Data never leaves the device
- ✅ **Infinitely scalable** - No server infrastructure required
- ✅ **Type-safe** - Full TypeScript support throughout
- ✅ **Framework-agnostic** - Core concepts work with any framework

### Architecture Benefits
- **Web Workers** prevent UI blocking during inference
- **TypeScript** provides compile-time safety and better DX
- **React Hooks** offer clean state management
- **Vite** enables fast development and optimized builds

### Next Steps
1. Train your own model and export to ONNX
2. Customize the UI for your use case
3. Add production monitoring and analytics
4. Consider implementing advanced features

The combination of ONNX Runtime Web, TypeScript, and modern web frameworks makes browser-based ML accessible and practical for production applications.
