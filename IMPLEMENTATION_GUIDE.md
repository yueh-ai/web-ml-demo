# Browser-Based Machine Learning Implementation Guide

This guide explains how to integrate client-side machine learning capabilities into your existing web application, regardless of your tech stack. The core concept is running ML models directly in the browser using ONNX Runtime Web, eliminating the need for server-side inference.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Architecture Overview](#architecture-overview)
3. [Model Preparation](#model-preparation)
4. [Core Implementation](#core-implementation)
5. [Framework Integration](#framework-integration)
6. [Build Tool Configuration](#build-tool-configuration)
7. [State Management Integration](#state-management-integration)
8. [Testing & CI/CD](#testing--cicd)
9. [Performance & Optimization](#performance--optimization)
10. [Production Considerations](#production-considerations)

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
│                  Your Existing Application               │
│                                                          │
│  ┌──────────────┐     ┌─────────────┐     ┌─────────┐  │
│  │ UI Component │ <-> │ ML Service  │ <-> │ Worker  │  │
│  │ (any framework)    │ (adapter)   │     │ Thread  │  │
│  └──────────────┘     └─────────────┘     └─────────┘  │
│                                                  │        │
│                                                  ▼        │
│                                            ┌─────────┐   │
│                                            │ ONNX    │   │
│                                            │ Runtime │   │
│                                            └─────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Model Preparation

### Step 1: Export Your Model to ONNX

The first step is converting your trained model to ONNX format. Here's how for common frameworks:

#### Python (scikit-learn, XGBoost, LightGBM)

```python
# model_export.py
import json
import numpy as np
from pathlib import Path

def export_sklearn_model(model, sample_input, output_dir):
    """Export scikit-learn compatible model to ONNX"""
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    # Define input type
    initial_type = [('float_input', FloatTensorType([None, sample_input.shape[1]]))]

    # Convert to ONNX
    onnx_model = to_onnx(
        model,
        initial_types=initial_type,
        target_opset=12,  # Use opset 12 for browser compatibility
        options={'zipmap': False}  # Return array instead of dict
    )

    # Save model
    output_path = Path(output_dir) / "model.onnx"
    with open(output_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    return onnx_model

def export_tensorflow_model(model, sample_input, output_dir):
    """Export TensorFlow/Keras model to ONNX"""
    import tf2onnx

    output_path = Path(output_dir) / "model.onnx"
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        output_path=str(output_path),
        opset=12
    )

    return model_proto

def export_pytorch_model(model, sample_input, output_dir):
    """Export PyTorch model to ONNX"""
    import torch

    model.eval()
    output_path = Path(output_dir) / "model.onnx"

    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
```

#### Generate Model Metadata

Create a metadata file that describes your model's inputs and outputs:

```python
def create_model_metadata(model_info, output_dir):
    """Create metadata file for the frontend"""
    metadata = {
        "modelInfo": {
            "name": model_info["name"],
            "version": model_info["version"],
            "type": model_info["type"],  # "regression", "classification", etc.
        },
        "inputs": {
            "features": model_info["feature_names"],
            "shape": model_info["input_shape"],
            "tensorName": model_info["input_tensor_name"],
        },
        "outputs": {
            "tensorName": model_info["output_tensor_name"],
            "shape": model_info["output_shape"],
            "type": model_info["output_type"],  # "probability", "value", etc.
        },
        # Optional: Include normalization parameters if needed
        "preprocessing": {
            "normalization": model_info.get("normalization", None)
        }
    }

    output_path = Path(output_dir) / "model-metadata.json"
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

### Step 2: Optimize Model Size (Optional)

For web deployment, smaller models load faster:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize_model_for_web(onnx_model_path, output_path):
    """Quantize model to reduce size"""
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True
    )
```

## Core Implementation

### Step 1: Install ONNX Runtime Web

```bash
# npm
npm install onnxruntime-web

# yarn
yarn add onnxruntime-web

# pnpm
pnpm add onnxruntime-web

# For TypeScript support
npm install --save-dev typescript @types/react @types/react-dom
```

### Step 2: Define TypeScript Types (Recommended)

For better type safety and clearer contracts between main thread and worker:

```typescript
// types/worker-messages.ts
// Model metadata types
export interface ModelMetadata {
  model_info: {
    type: string;
    r2_score: number;
    training_samples: number;
  };
  feature_ranges: Record<string, FeatureRange>;
}

export interface FeatureRange {
  min: number;
  max: number;
  default: number;
  step: number;
}

export type FeatureData = Record<string, number>;

// Worker message types
export interface LoadModelMessage {
  event: 'load';
  meta: ModelMetadata;
}

export interface PredictMessage {
  event: 'predict';
  features: FeatureData;
}

export type WorkerIncomingMessage = LoadModelMessage | PredictMessage;

// Messages from worker to main thread
export interface ModelLoadedMessage {
  event: 'model_loaded';
  success: boolean;
  error?: string;
}

export interface PredictionMessage {
  event: 'prediction';
  success: boolean;
  value?: number;
  error?: string;
}

export type WorkerOutgoingMessage = ModelLoadedMessage | PredictionMessage;
```

### Step 3: Create the Web Worker

TypeScript version (recommended):

```typescript
// workers/predict.worker.ts
import * as ort from 'onnxruntime-web';
import type { WorkerIncomingMessage, WorkerOutgoingMessage, ModelMetadata, FeatureData } from '../types/worker-messages';

// Configure ONNX Runtime
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let session: ort.InferenceSession | null = null;
let meta: ModelMetadata | null = null;

async function loadModel(metadata: ModelMetadata) {
  try {
    meta = metadata;
    
    // Create inference session
    session = await ort.InferenceSession.create('/models/xgb_california_housing.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });

    console.log('Model loaded successfully');
    
    const message: WorkerOutgoingMessage = {
      event: 'model_loaded',
      success: true
    };
    self.postMessage(message);
  } catch (error) {
    console.error('Failed to load model:', error);
    const message: WorkerOutgoingMessage = {
      event: 'model_loaded',
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
    self.postMessage(message);
  }
}

async function predict(features: FeatureData) {
  if (!session || !meta) {
    const message: WorkerOutgoingMessage = {
      event: 'prediction',
      success: false,
      error: 'Model not loaded'
    };
    self.postMessage(message);
    return;
  }

  try {
    // Extract feature values in the correct order
    const featureNames = Object.keys(meta.feature_ranges);
    const inputArray = featureNames.map(name => features[name]);
    
    // Create input tensor
    const inputTensor = new ort.Tensor('float32', Float32Array.from(inputArray), [1, inputArray.length]);
    
    // Run inference
    const feeds = { 'input': inputTensor };
    const results = await session.run(feeds);
    
    // Extract prediction
    const outputTensor = results['variable'];
    const prediction = outputTensor.data[0] as number;
    
    const message: WorkerOutgoingMessage = {
      event: 'prediction',
      success: true,
      value: prediction
    };
    self.postMessage(message);
  } catch (error) {
    console.error('Prediction failed:', error);
    const message: WorkerOutgoingMessage = {
      event: 'prediction',
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
    self.postMessage(message);
  }
}

// Handle messages from main thread
self.onmessage = async ({ data }: MessageEvent<WorkerIncomingMessage>) => {
  switch (data.event) {
    case 'load':
      await loadModel(data.meta);
      break;
    case 'predict':
      await predict(data.features);
      break;
  }
};
```

JavaScript version:

```javascript
// ml-worker.js
import * as ort from "onnxruntime-web";

// Configure ONNX Runtime
ort.env.wasm.wasmPaths = "/"; // Adjust based on your setup

class MLWorker {
  constructor() {
    this.session = null;
    this.metadata = null;
  }

  async loadModel(modelUrl, metadataUrl) {
    try {
      // Load metadata
      const metadataResponse = await fetch(metadataUrl);
      this.metadata = await metadataResponse.json();

      // Create inference session
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });

      return { success: true, metadata: this.metadata };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  async predict(inputData) {
    if (!this.session) {
      throw new Error("Model not loaded");
    }

    try {
      // Create input tensor
      const inputTensor = new ort.Tensor(
        "float32",
        new Float32Array(inputData),
        [1, inputData.length]
      );

      // Run inference
      const feeds = { [this.metadata.inputs.tensorName]: inputTensor };
      const results = await this.session.run(feeds);

      // Extract output
      const output = results[this.metadata.outputs.tensorName];
      return Array.from(output.data);
    } catch (error) {
      throw new Error(`Prediction failed: ${error.message}`);
    }
  }
}

// Worker message handler
const worker = new MLWorker();

self.addEventListener("message", async (event) => {
  const { type, id, payload } = event.data;

  try {
    let result;

    switch (type) {
      case "LOAD_MODEL":
        result = await worker.loadModel(payload.modelUrl, payload.metadataUrl);
        break;

      case "PREDICT":
        result = await worker.predict(payload.inputData);
        break;

      default:
        throw new Error(`Unknown message type: ${type}`);
    }

    self.postMessage({ type: "SUCCESS", id, result });
  } catch (error) {
    self.postMessage({
      type: "ERROR",
      id,
      error: error.message,
    });
  }
});
```

### Step 4: Create React Hooks for Model Management

TypeScript version (recommended):

```typescript
// hooks/useModelLoader.ts
import { useEffect, useState, useRef } from 'react';
import type { ModelMetadata, WorkerOutgoingMessage } from '../types/worker-messages';

interface UseModelLoaderReturn {
  meta: ModelMetadata | null;
  ready: boolean;
  loading: boolean;
  error: string | null;
  worker: Worker | null;
}

export function useModelLoader(): UseModelLoaderReturn {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    async function initModel() {
      try {
        // Fetch metadata
        const response = await fetch('/models/xgb_california_housing_metadata.json');
        if (!response.ok) {
          throw new Error('Failed to load model metadata');
        }
        
        const metadata = await response.json() as ModelMetadata;
        setMeta(metadata);
        
        // Import worker with Vite's ?worker syntax
        const PredictWorker = await import('../workers/predict.worker.ts?worker');
        const worker = new PredictWorker.default();
        workerRef.current = worker;
        
        // Setup message handler
        const handleMessage = ({ data }: MessageEvent<WorkerOutgoingMessage>) => {
          if (data.event === 'model_loaded') {
            if (data.success) {
              setReady(true);
              setLoading(false);
            } else {
              setError(data.error || 'Failed to load model');
              setLoading(false);
            }
          }
        };
        
        worker.addEventListener('message', handleMessage);
        
        // Load model
        worker.postMessage({ event: 'load', meta: metadata });
        
        return () => {
          worker.removeEventListener('message', handleMessage);
        };
      } catch (err) {
        console.error('Failed to initialize model:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        setLoading(false);
      }
    }
    
    initModel();
    
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);
  
  return { meta, ready, loading, error, worker: workerRef.current };
}

// hooks/usePredict.ts
import { useEffect, useState, useRef } from 'react';
import type { ModelMetadata, FeatureData, WorkerOutgoingMessage } from '../types/worker-messages';

interface UsePredictParams {
  worker: Worker | null;
  meta: ModelMetadata | null;
  features: FeatureData | null;
}

interface UsePredictReturn {
  prediction: number | null;
  predicting: boolean;
  error: string | null;
}

export function usePredict({ worker, meta, features }: UsePredictParams): UsePredictReturn {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!worker || !meta || !features) return;

    // Clear previous timeout
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    // Debounce predictions for smooth UI
    debounceRef.current = setTimeout(() => {
      setPredicting(true);
      setError(null);
      
      // Send prediction request
      worker.postMessage({ 
        event: 'predict', 
        features 
      });
    }, 50); // 50ms debounce

    // Cleanup
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
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

    return () => {
      worker.removeEventListener('message', handleMessage);
    };
  }, [worker]);

  return { 
    prediction, 
    predicting, 
    error 
  };
}
```

JavaScript version:

```javascript
// ml-service.js
export class MLService {
  constructor() {
    this.worker = null;
    this.messageId = 0;
    this.pendingMessages = new Map();
    this.modelMetadata = null;
  }

  async initialize(modelUrl, metadataUrl) {
    // Create worker
    this.worker = new Worker(new URL("./ml-worker.js", import.meta.url), {
      type: "module",
    });

    // Setup message handler
    this.worker.addEventListener("message", (event) => {
      const { type, id, result, error } = event.data;
      const pending = this.pendingMessages.get(id);

      if (pending) {
        if (type === "SUCCESS") {
          pending.resolve(result);
        } else {
          pending.reject(new Error(error));
        }
        this.pendingMessages.delete(id);
      }
    });

    // Load model
    const loadResult = await this.sendMessage("LOAD_MODEL", {
      modelUrl,
      metadataUrl,
    });

    if (loadResult.success) {
      this.modelMetadata = loadResult.metadata;
      return loadResult;
    } else {
      throw new Error(loadResult.error);
    }
  }

  async predict(inputData) {
    if (!this.worker) {
      throw new Error("ML Service not initialized");
    }

    return this.sendMessage("PREDICT", { inputData });
  }

  sendMessage(type, payload) {
    return new Promise((resolve, reject) => {
      const id = this.messageId++;
      this.pendingMessages.set(id, { resolve, reject });
      this.worker.postMessage({ type, id, payload });
    });
  }

  dispose() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.pendingMessages.clear();
  }
}
```

### Step 5: React Component Integration

Here's how to create a React component that uses the hooks:

```typescript
// components/DemoPredictor.tsx
import { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Slider,
  Grid,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import { useModelLoader } from '../hooks/useModelLoader';
import { usePredict } from '../hooks/usePredict';
import type { FeatureData } from '../types/worker-messages';

export default function DemoPredictor() {
  const { meta, ready, loading, error: loadError, worker } = useModelLoader();
  
  // Initialize features with default values
  const [features, setFeatures] = useState<FeatureData | null>(null);

  // Initialize features when metadata is loaded
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
    worker,
    meta,
    features
  });

  const handleSliderChange = (featureName: string) => (_: Event, value: number | number[]) => {
    setFeatures(prev => ({
      ...prev!,
      [featureName]: value as number
    }));
  };

  if (loading) {
    return (
      <Container maxWidth="md">
        <Paper sx={{ p: 4, mt: 4 }}>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={300}>
            <CircularProgress size={60} />
          </Box>
        </Paper>
      </Container>
    );
  }

  if (loadError) {
    return (
      <Container maxWidth="md">
        <Paper sx={{ p: 4, mt: 4 }}>
          <Alert severity="error">
            Failed to load model: {loadError}
          </Alert>
        </Paper>
      </Container>
    );
  }

  if (!ready || !features || !meta) {
    return null;
  }

  return (
    <Container maxWidth="md">
      <Paper sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          California House Price Prediction
        </Typography>
        
        <Typography variant="body1" color="textSecondary" paragraph>
          Adjust the sliders below to see real-time predictions using an XGBoost model
          running entirely in your browser.
        </Typography>

        <Grid container spacing={3}>
          {Object.entries(meta.feature_ranges).map(([name, range]) => (
            <Grid item xs={12} sm={6} key={name}>
              <Box mb={3}>
                <Typography gutterBottom>
                  {name}: <strong>{features[name].toFixed(range.step < 1 ? 1 : 0)}</strong>
                </Typography>
                <Slider
                  value={features[name]}
                  min={range.min}
                  max={range.max}
                  step={range.step}
                  onChange={handleSliderChange(name)}
                  valueLabelDisplay="auto"
                  marks={[
                    { value: range.min, label: String(range.min) },
                    { value: range.max, label: String(range.max) }
                  ]}
                />
              </Box>
            </Grid>
          ))}
        </Grid>

        {predictError && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Prediction error: {predictError}
          </Alert>
        )}

        <Box mt={4} p={3} bgcolor="primary.main" color="primary.contrastText" borderRadius={1} textAlign="center">
          <Typography variant="h6">
            Predicted House Price
          </Typography>
          <Typography variant="h3">
            {predicting ? (
              <CircularProgress size={40} color="inherit" />
            ) : prediction !== null ? (
              `$${(prediction * 100000).toLocaleString('en-US', { 
                minimumFractionDigits: 0,
                maximumFractionDigits: 0 
              })}`
            ) : (
              '—'
            )}
          </Typography>
          <Typography variant="caption">
            (Price in hundreds of thousands of dollars)
          </Typography>
        </Box>

        <Box mt={3}>
          <Typography variant="caption" color="textSecondary">
            Model: {meta.model_info.type} | 
            R² Score: {meta.model_info.r2_score} | 
            Training Samples: {meta.model_info.training_samples.toLocaleString()}
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
}

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

### Vite Configuration (Recommended for TypeScript)

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      // Required for SharedArrayBuffer
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    }
  },
  worker: {
    format: 'es'
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
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
    "noFallthroughCasesInSwitch": true,
    "allowJs": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### Worker Import Types

```typescript
// types/worker-imports.d.ts
declare module '*?worker' {
  const workerConstructor: {
    new (): Worker;
  };
  export default workerConstructor;
}
```

### Webpack Configuration

```javascript
// webpack.config.js
module.exports = {
  // ... your existing config
  module: {
    rules: [
      // ... existing rules
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.onnx$/,
        type: "asset/resource",
        generator: {
          filename: "models/[name][ext]",
        },
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
    fallback: {
      path: false,
      fs: false,
    },
  },
  // Copy WASM files
  plugins: [
    new CopyWebpackPlugin({
      patterns: [
        {
          from: "node_modules/onnxruntime-web/dist/*.wasm",
          to: "[name][ext]",
        },
      ],
    }),
  ],
};
```

## State Management Integration

### Redux Integration

```javascript
// store/mlSlice.js
import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { MLService } from "../services/ml-service";

const mlService = new MLService();

export const initializeModel = createAsyncThunk(
  "ml/initialize",
  async ({ modelUrl, metadataUrl }) => {
    const result = await mlService.initialize(modelUrl, metadataUrl);
    return result.metadata;
  }
);

export const predict = createAsyncThunk("ml/predict", async (inputData) => {
  return await mlService.predict(inputData);
});

const mlSlice = createSlice({
  name: "ml",
  initialState: {
    isLoading: true,
    metadata: null,
    prediction: null,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(initializeModel.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(initializeModel.fulfilled, (state, action) => {
        state.isLoading = false;
        state.metadata = action.payload;
      })
      .addCase(initializeModel.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message;
      })
      .addCase(predict.fulfilled, (state, action) => {
        state.prediction = action.payload;
      });
  },
});

export default mlSlice.reducer;
```

### MobX Integration

```javascript
// stores/MLStore.js
import { makeAutoObservable, runInAction } from "mobx";
import { MLService } from "../services/ml-service";

class MLStore {
  mlService = new MLService();
  isLoading = true;
  metadata = null;
  prediction = null;
  error = null;

  constructor() {
    makeAutoObservable(this);
  }

  async initialize(modelUrl, metadataUrl) {
    try {
      const result = await this.mlService.initialize(modelUrl, metadataUrl);
      runInAction(() => {
        this.metadata = result.metadata;
        this.isLoading = false;
      });
    } catch (error) {
      runInAction(() => {
        this.error = error.message;
        this.isLoading = false;
      });
    }
  }

  async predict(inputData) {
    const result = await this.mlService.predict(inputData);
    runInAction(() => {
      this.prediction = result;
    });
  }

  dispose() {
    this.mlService.dispose();
  }
}

export default new MLStore();
```

### Zustand Integration

```javascript
// stores/useMLStore.js
import create from "zustand";
import { MLService } from "../services/ml-service";

const mlService = new MLService();

export const useMLStore = create((set, get) => ({
  isLoading: true,
  metadata: null,
  prediction: null,
  error: null,

  initialize: async (modelUrl, metadataUrl) => {
    try {
      const result = await mlService.initialize(modelUrl, metadataUrl);
      set({
        metadata: result.metadata,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
    }
  },

  predict: async (inputData) => {
    const result = await mlService.predict(inputData);
    set({ prediction: result });
  },

  cleanup: () => {
    mlService.dispose();
  },
}));
```

## Testing & CI/CD

### Unit Testing

```javascript
// __tests__/ml-service.test.js
import { MLService } from "../ml-service";

// Mock worker
class MockWorker {
  constructor() {
    this.onmessage = null;
    this.postMessage = jest.fn();
  }

  addEventListener(event, handler) {
    if (event === "message") {
      this.onmessage = handler;
    }
  }

  terminate() {}
}

global.Worker = MockWorker;

describe("MLService", () => {
  let mlService;

  beforeEach(() => {
    mlService = new MLService();
  });

  afterEach(() => {
    mlService.dispose();
  });

  test("initializes model successfully", async () => {
    const promise = mlService.initialize("/model.onnx", "/metadata.json");

    // Simulate successful response
    mlService.worker.onmessage({
      data: {
        type: "SUCCESS",
        id: 0,
        result: { success: true, metadata: { test: "data" } },
      },
    });

    const result = await promise;
    expect(result.success).toBe(true);
    expect(mlService.modelMetadata).toEqual({ test: "data" });
  });
});
```

### Integration Testing

```javascript
// __tests__/ml-integration.test.js
import puppeteer from "puppeteer";

describe("ML Integration", () => {
  let browser;
  let page;

  beforeAll(async () => {
    browser = await puppeteer.launch();
    page = await browser.newPage();
  });

  afterAll(async () => {
    await browser.close();
  });

  test("loads model and makes prediction", async () => {
    await page.goto("http://localhost:3000");

    // Wait for model to load
    await page.waitForSelector('[data-testid="model-loaded"]');

    // Input values
    await page.type('[name="feature1"]', "5.0");
    await page.type('[name="feature2"]', "3.5");

    // Submit prediction
    await page.click('[data-testid="predict-button"]');

    // Check result
    await page.waitForSelector('[data-testid="prediction-result"]');
    const result = await page.$eval(
      '[data-testid="prediction-result"]',
      (el) => el.textContent
    );

    expect(result).toMatch(/\d+\.\d+/);
  });
});
```

### CI/CD Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on: [push, pull_request]

jobs:
  test-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Test model export
        run: |
          python scripts/test_model_export.py

      - name: Validate ONNX model
        run: |
          python -c "import onnx; onnx.checker.check_model('model.onnx')"

      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: ml-models
          path: |
            model.onnx
            model-metadata.json

  test-frontend:
    needs: test-model
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Download model artifacts
        uses: actions/download-artifact@v3
        with:
          name: ml-models
          path: public/models

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: "18"

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Build application
        run: npm run build
```

## Performance & Optimization

### Model Loading Optimization

```javascript
// Preload model during idle time
if ("requestIdleCallback" in window) {
  requestIdleCallback(() => {
    mlService.initialize("/models/model.onnx", "/models/metadata.json");
  });
}

// Cache model in IndexedDB
class ModelCache {
  async saveModel(modelData, metadata) {
    const db = await this.openDB();
    const tx = db.transaction(["models"], "readwrite");
    await tx.objectStore("models").put({
      id: "primary",
      modelData,
      metadata,
      timestamp: Date.now(),
    });
  }

  async loadModel() {
    const db = await this.openDB();
    const tx = db.transaction(["models"], "readonly");
    return await tx.objectStore("models").get("primary");
  }

  async openDB() {
    return await openDB("ml-models", 1, {
      upgrade(db) {
        db.createObjectStore("models", { keyPath: "id" });
      },
    });
  }
}
```

### Inference Optimization

```javascript
// Batch predictions for efficiency
class BatchPredictor {
  constructor(mlService, batchSize = 32, delay = 100) {
    this.mlService = mlService;
    this.batchSize = batchSize;
    this.delay = delay;
    this.queue = [];
    this.timer = null;
  }

  predict(inputData) {
    return new Promise((resolve, reject) => {
      this.queue.push({ inputData, resolve, reject });

      if (this.queue.length >= this.batchSize) {
        this.processBatch();
      } else {
        this.scheduleBatch();
      }
    });
  }

  scheduleBatch() {
    if (this.timer) return;

    this.timer = setTimeout(() => {
      this.processBatch();
    }, this.delay);
  }

  async processBatch() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    const batch = this.queue.splice(0, this.batchSize);
    if (batch.length === 0) return;

    try {
      // Combine inputs
      const batchInput = batch.map((item) => item.inputData).flat();

      // Run batch prediction
      const results = await this.mlService.predictBatch(
        batchInput,
        batch.length
      );

      // Distribute results
      results.forEach((result, i) => {
        batch[i].resolve(result);
      });
    } catch (error) {
      batch.forEach((item) => item.reject(error));
    }
  }
}
```

### Memory Management

```javascript
// Dispose tensors properly
class TensorManager {
  constructor() {
    this.tensors = new Set();
  }

  createTensor(data, dims) {
    const tensor = new ort.Tensor("float32", data, dims);
    this.tensors.add(tensor);
    return tensor;
  }

  async runInference(session, feeds) {
    try {
      const results = await session.run(feeds);
      return results;
    } finally {
      // Clean up input tensors
      for (const tensor of Object.values(feeds)) {
        tensor.dispose();
        this.tensors.delete(tensor);
      }
    }
  }

  dispose() {
    for (const tensor of this.tensors) {
      tensor.dispose();
    }
    this.tensors.clear();
  }
}
```

## Production Considerations

### Error Handling & Fallbacks

```javascript
class RobustMLService extends MLService {
  constructor(options = {}) {
    super();
    this.maxRetries = options.maxRetries || 3;
    this.fallbackUrl = options.fallbackUrl;
  }

  async initialize(modelUrl, metadataUrl, retryCount = 0) {
    try {
      return await super.initialize(modelUrl, metadataUrl);
    } catch (error) {
      if (retryCount < this.maxRetries) {
        console.warn(`Model loading failed, retry ${retryCount + 1}`);
        return this.initialize(modelUrl, metadataUrl, retryCount + 1);
      }

      // Fallback to server-side inference
      if (this.fallbackUrl) {
        console.warn("Falling back to server-side inference");
        this.useFallback = true;
        return {
          success: true,
          metadata: await this.fetchMetadata(metadataUrl),
        };
      }

      throw error;
    }
  }

  async predict(inputData) {
    if (this.useFallback) {
      return this.serverPredict(inputData);
    }
    return super.predict(inputData);
  }

  async serverPredict(inputData) {
    const response = await fetch(this.fallbackUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: inputData }),
    });
    const data = await response.json();
    return data.prediction;
  }
}
```

### Monitoring & Analytics

```javascript
class MLMetrics {
  constructor(analyticsProvider) {
    this.analytics = analyticsProvider;
    this.metrics = {
      loadTime: 0,
      inferenceCount: 0,
      totalInferenceTime: 0,
      errors: [],
    };
  }

  async trackModelLoad(startTime) {
    this.metrics.loadTime = Date.now() - startTime;

    await this.analytics.track("ml_model_loaded", {
      load_time_ms: this.metrics.loadTime,
      model_size_kb: await this.getModelSize(),
      browser: navigator.userAgent,
    });
  }

  async trackInference(startTime, inputSize, success = true) {
    const inferenceTime = Date.now() - startTime;
    this.metrics.inferenceCount++;
    this.metrics.totalInferenceTime += inferenceTime;

    await this.analytics.track("ml_inference", {
      inference_time_ms: inferenceTime,
      input_size: inputSize,
      success,
      average_time_ms:
        this.metrics.totalInferenceTime / this.metrics.inferenceCount,
    });
  }

  trackError(error, context) {
    this.metrics.errors.push({ error, context, timestamp: Date.now() });

    this.analytics.track("ml_error", {
      error_message: error.message,
      error_stack: error.stack,
      context,
    });
  }
}
```

### Security Considerations

```javascript
class SecureMLService extends MLService {
  validateInput(inputData) {
    // Validate input shape
    if (
      !Array.isArray(inputData) ||
      inputData.length !== this.modelMetadata.inputs.features.length
    ) {
      throw new Error("Invalid input shape");
    }

    // Validate input values
    for (const value of inputData) {
      if (typeof value !== "number" || !isFinite(value)) {
        throw new Error("Invalid input value");
      }
    }

    // Check for potential attacks (e.g., extremely large values)
    const MAX_VALUE = 1e6;
    if (inputData.some((v) => Math.abs(v) > MAX_VALUE)) {
      throw new Error("Input values out of range");
    }

    return true;
  }

  async predict(inputData) {
    this.validateInput(inputData);
    return super.predict(inputData);
  }
}
```

### Progressive Enhancement

```javascript
// Feature detection and graceful degradation
class MLFeatureDetection {
  static isSupported() {
    // Check for required features
    const hasWebAssembly = typeof WebAssembly !== "undefined";
    const hasWorker = typeof Worker !== "undefined";
    const hasFloat32Array = typeof Float32Array !== "undefined";

    return hasWebAssembly && hasWorker && hasFloat32Array;
  }

  static async checkPerformance() {
    // Simple performance test
    const testSize = 1000000;
    const data = new Float32Array(testSize);

    const start = performance.now();
    for (let i = 0; i < testSize; i++) {
      data[i] = Math.random();
    }
    const elapsed = performance.now() - start;

    // If basic operations are too slow, fall back
    return elapsed < 100; // 100ms threshold
  }
}

// Usage
async function initializeMLFeature() {
  if (!MLFeatureDetection.isSupported()) {
    console.log("Browser ML not supported, using server fallback");
    return new ServerMLService();
  }

  const hasGoodPerformance = await MLFeatureDetection.checkPerformance();
  if (!hasGoodPerformance) {
    console.log("Device too slow for client-side ML");
    return new ServerMLService();
  }

  return new MLService();
}
```

## TypeScript Migration Benefits

When migrating to TypeScript, you gain several advantages:

1. **Type-Safe Message Passing**: The worker communication is now fully typed, preventing runtime errors from mismatched message formats
2. **Better IDE Support**: Auto-completion and inline documentation for all ML-related functions
3. **Compile-Time Error Detection**: Catch type mismatches before runtime
4. **Self-Documenting Code**: Types serve as inline documentation for the expected data structures

## Summary

This guide provides a complete framework for integrating browser-based machine learning into any web application with TypeScript support. The key steps are:

1. **Export your model** to ONNX format with appropriate metadata
2. **Define TypeScript types** for worker messages and data structures
3. **Create a typed Web Worker** to handle model loading and inference
4. **Build React hooks** to manage worker communication and state
5. **Create React components** that use the hooks for UI
6. **Configure your build tools** (Vite recommended) for TypeScript and workers
7. **Add proper error handling** and type validation

The TypeScript implementation provides:

- **Type Safety** - Compile-time checking for all worker communications
- **Better Developer Experience** - IDE support with auto-completion
- **Maintainability** - Clear contracts between components
- **Framework Integration** - Works seamlessly with React and TypeScript
- **Production Ready** - Includes error handling and proper typing
- **Performance Optimized** - Uses web workers for non-blocking inference

Remember to:
- Use strict TypeScript mode for maximum type safety
- Test thoroughly across different browsers and devices
- Monitor WebAssembly performance on various hardware
- Consider fallbacks for browsers that don't support required features
