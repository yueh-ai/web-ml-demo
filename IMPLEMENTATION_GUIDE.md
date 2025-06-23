# Browser-Based Machine Learning Implementation Guide

This guide provides comprehensive instructions for implementing client-side machine learning in web applications. Based on a production-ready demo that runs XGBoost models entirely in the browser using ONNX Runtime Web.

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Development](#model-development)
4. [Web Implementation](#web-implementation)
5. [Integration Steps](#integration-steps)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Deployment](#deployment)

## Overview

This implementation enables running machine learning models directly in web browsers without server-side inference. Key benefits:
- **Zero latency**: No network calls for predictions
- **Privacy**: Data never leaves the user's device
- **Scalability**: No server infrastructure needed for inference
- **Offline capability**: Works without internet connection

### Technology Stack
- **Model Training**: Python, XGBoost, scikit-learn
- **Model Format**: ONNX (Open Neural Network Exchange)
- **Frontend**: React, Material-UI
- **ML Runtime**: ONNX Runtime Web (WebAssembly)
- **Build Tool**: Vite

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Model Development                     │
│  Python Script → Train Model → Export ONNX → Metadata   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Web Application                       │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ React App   │ ←→ │ Web Worker   │ ←→ │ ONNX      │  │
│  │ (UI Thread) │    │ (Background) │    │ Runtime   │  │
│  └─────────────┘    └──────────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Model Development

### 1. Setup Python Environment

Create a `pyproject.toml` for dependency management:

```toml
[project]
name = "model-dev"
version = "0.1.0"
description = "ML model development for browser deployment"
requires-python = ">=3.11"
dependencies = [
    "xgboost>=2.1.3",
    "scikit-learn>=1.5.2",
    "skl2onnx>=1.18.0",
    "onnxmltools>=1.13.0",
    "onnxruntime>=1.20.1",
    "numpy>=1.26.4",
    "pandas>=2.2.3",
]
```

### 2. Train and Export Model

Complete model training and export script (`build_model.py`):

```python
import json
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
from skl2onnx import to_onnx
from onnxmltools.utils import save_model
import onnxruntime as ort

def train_model():
    """Train XGBoost model on California Housing dataset"""
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    
    # Use subset for faster training (adjust as needed)
    X = data.data[:10000]
    y = data.target[:10000]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model R² score: {r2:.3f}")
    
    return model, data, r2

def export_to_onnx(model, sample_input, output_path):
    """Convert sklearn model to ONNX format"""
    print("Converting to ONNX format...")
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, sample_input.shape[1]]))]
    
    # Convert to ONNX
    onnx_model = to_onnx(
        model, 
        initial_types=initial_type,
        target_opset=12,  # Compatible with most browsers
        options={'zipmap': False}  # Return array instead of dict
    )
    
    # Save model
    save_model(onnx_model, output_path)
    print(f"Model saved to {output_path}")
    
    # Verify the model
    session = ort.InferenceSession(str(output_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    return input_name, output_name

def create_metadata(data, r2_score, input_name, output_name, output_dir):
    """Create metadata file for frontend"""
    feature_names = data.feature_names
    
    # Calculate feature statistics for UI sliders
    feature_stats = {}
    for i, name in enumerate(feature_names):
        values = data.data[:, i]
        feature_stats[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    
    metadata = {
        "model": {
            "name": "California Housing Price Predictor",
            "type": "xgboost",
            "task": "regression",
            "performance": {
                "r2_score": r2_score,
                "metric": "R²"
            }
        },
        "inputs": {
            "features": feature_names,
            "tensorName": input_name,
            "shape": [-1, len(feature_names)]
        },
        "outputs": {
            "tensorName": output_name,
            "type": "price",
            "unit": "$1000s"
        },
        "featureStats": feature_stats
    }
    
    # Save metadata
    metadata_path = output_dir / "model-package.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")

def main():
    # Setup paths
    output_dir = Path("../web/public")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    model, data, r2 = train_model()
    
    # Export to ONNX
    sample_input = data.data[:1]  # Single sample for type inference
    model_path = output_dir / "model.onnx"
    input_name, output_name = export_to_onnx(model, sample_input, model_path)
    
    # Create metadata
    create_metadata(data, r2, input_name, output_name, output_dir)
    
    # Print model size
    model_size = model_path.stat().st_size / 1024
    print(f"\nModel size: {model_size:.1f} KB")
    print("Build complete!")

if __name__ == "__main__":
    main()
```

### 3. Model Optimization Tips

For smaller model sizes:

```python
# Option 1: Quantization
from onnxruntime.quantization import quantize_dynamic

quantized_model_path = "model_quantized.onnx"
quantize_dynamic(
    model_input="model.onnx",
    model_output=quantized_model_path,
    weight_type=QuantType.QUInt8
)

# Option 2: Use lighter models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Linear model (smallest)
model = LinearRegression()

# Or Random Forest with limited trees
model = RandomForestRegressor(n_estimators=10, max_depth=3)
```

## Web Implementation

### 1. Project Setup

Initialize React project with Vite:

```bash
npm create vite@latest web -- --template react
cd web
npm install @mui/material @emotion/react @emotion/styled onnxruntime-web
```

### 2. Configure Vite

Create `vite.config.js`:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    rollupOptions: {
      output: {
        format: 'es'
      }
    }
  },
  assetsInclude: ['**/*.onnx']
})
```

### 3. Create Web Worker

Create `src/workers/predict.worker.js`:

```javascript
import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime for web
ort.env.wasm.wasmPaths = '/';

let session = null;
let inputName = null;
let outputName = null;

// Handle messages from main thread
self.addEventListener('message', async (event) => {
  const { type, payload } = event.data;
  
  try {
    switch (type) {
      case 'LOAD_MODEL':
        await loadModel(payload.modelPath, payload.metadata);
        break;
        
      case 'PREDICT':
        const prediction = await predict(payload.features);
        self.postMessage({ 
          type: 'PREDICTION_RESULT', 
          payload: { prediction } 
        });
        break;
        
      default:
        throw new Error(`Unknown message type: ${type}`);
    }
  } catch (error) {
    self.postMessage({ 
      type: 'ERROR', 
      payload: { 
        message: error.message,
        stack: error.stack 
      } 
    });
  }
});

async function loadModel(modelPath, metadata) {
  try {
    console.log('Loading ONNX model...');
    
    // Create inference session
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    
    // Store tensor names
    inputName = metadata.inputs.tensorName;
    outputName = metadata.outputs.tensorName;
    
    console.log('Model loaded successfully');
    self.postMessage({ type: 'MODEL_LOADED' });
    
  } catch (error) {
    console.error('Failed to load model:', error);
    throw error;
  }
}

async function predict(features) {
  if (!session) {
    throw new Error('Model not loaded');
  }
  
  try {
    // Convert features to Float32Array
    const inputData = new Float32Array(features);
    
    // Create input tensor
    const inputTensor = new ort.Tensor('float32', inputData, [1, features.length]);
    
    // Run inference
    const feeds = { [inputName]: inputTensor };
    const results = await session.run(feeds);
    
    // Extract prediction
    const output = results[outputName];
    const prediction = output.data[0];
    
    return prediction;
    
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
}
```

### 4. Create Custom Hooks

Create `src/hooks/useModelLoader.js`:

```javascript
import { useState, useEffect, useRef } from 'react';

export function useModelLoader() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const workerRef = useRef(null);

  useEffect(() => {
    let mounted = true;

    async function initializeModel() {
      try {
        // Fetch model metadata
        const response = await fetch('/model-package.json');
        if (!response.ok) {
          throw new Error('Failed to fetch model metadata');
        }
        
        const modelMetadata = await response.json();
        
        if (!mounted) return;
        
        setMetadata(modelMetadata);
        
        // Create worker
        const worker = new Worker(
          new URL('../workers/predict.worker.js', import.meta.url),
          { type: 'module' }
        );
        
        // Setup message handler
        worker.addEventListener('message', (event) => {
          const { type, payload } = event.data;
          
          if (type === 'MODEL_LOADED') {
            if (mounted) {
              setIsLoading(false);
            }
          } else if (type === 'ERROR') {
            if (mounted) {
              setError(new Error(payload.message));
              setIsLoading(false);
            }
          }
        });
        
        // Load model in worker
        worker.postMessage({
          type: 'LOAD_MODEL',
          payload: {
            modelPath: '/model.onnx',
            metadata: modelMetadata
          }
        });
        
        workerRef.current = worker;
        
      } catch (err) {
        if (mounted) {
          setError(err);
          setIsLoading(false);
        }
      }
    }

    initializeModel();

    return () => {
      mounted = false;
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  return { isLoading, error, metadata, worker: workerRef.current };
}
```

Create `src/hooks/usePredict.js`:

```javascript
import { useState, useEffect, useRef } from 'react';

export function usePredict(worker, features) {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (!worker || !features) return;

    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Debounce predictions
    timeoutRef.current = setTimeout(() => {
      predict();
    }, 50);

    async function predict() {
      setIsLoading(true);
      setError(null);

      const messageHandler = (event) => {
        const { type, payload } = event.data;
        
        if (type === 'PREDICTION_RESULT') {
          setPrediction(payload.prediction);
          setIsLoading(false);
        } else if (type === 'ERROR') {
          setError(new Error(payload.message));
          setIsLoading(false);
        }
      };

      worker.addEventListener('message', messageHandler);

      // Send prediction request
      worker.postMessage({
        type: 'PREDICT',
        payload: { features }
      });

      // Cleanup
      return () => {
        worker.removeEventListener('message', messageHandler);
      };
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [worker, features]);

  return { prediction, isLoading, error };
}
```

### 5. Create UI Components

Create `src/components/DemoPredictor.jsx`:

```javascript
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  Grid,
  CircularProgress,
  Alert,
  Chip
} from '@mui/material';
import { useModelLoader } from '../hooks/useModelLoader';
import { usePredict } from '../hooks/usePredict';

export function DemoPredictor() {
  const { isLoading: modelLoading, error: modelError, metadata, worker } = useModelLoader();
  const [features, setFeatures] = useState(null);
  const { prediction, isLoading: predicting } = usePredict(worker, features);

  // Initialize features when metadata loads
  React.useEffect(() => {
    if (metadata && !features) {
      const initialFeatures = {};
      metadata.inputs.features.forEach((feature) => {
        initialFeatures[feature] = metadata.featureStats[feature].mean;
      });
      setFeatures(initialFeatures);
    }
  }, [metadata, features]);

  if (modelLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
        <Typography variant="body1" sx={{ ml: 2 }}>
          Loading ML model...
        </Typography>
      </Box>
    );
  }

  if (modelError) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load model: {modelError.message}
      </Alert>
    );
  }

  if (!metadata || !features) return null;

  const handleFeatureChange = (featureName, value) => {
    setFeatures(prev => ({
      ...prev,
      [featureName]: value
    }));
  };

  const formatFeatureName = (name) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <Card sx={{ maxWidth: 800, mx: 'auto', my: 4 }}>
      <CardContent>
        <Typography variant="h4" gutterBottom>
          {metadata.model.name}
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Chip 
            label={`Model: ${metadata.model.type}`} 
            size="small" 
            sx={{ mr: 1 }} 
          />
          <Chip 
            label={`${metadata.model.performance.metric}: ${metadata.model.performance.r2_score.toFixed(3)}`} 
            size="small" 
            color="primary" 
          />
        </Box>

        <Grid container spacing={3}>
          {metadata.inputs.features.map((feature) => {
            const stats = metadata.featureStats[feature];
            const value = features[feature];

            return (
              <Grid item xs={12} sm={6} key={feature}>
                <Typography gutterBottom>
                  {formatFeatureName(feature)}: {value.toFixed(2)}
                </Typography>
                <Slider
                  value={value}
                  onChange={(_, newValue) => handleFeatureChange(feature, newValue)}
                  min={stats.min}
                  max={stats.max}
                  step={(stats.max - stats.min) / 100}
                  valueLabelDisplay="auto"
                />
              </Grid>
            );
          })}
        </Grid>

        <Box sx={{ mt: 4, p: 3, bgcolor: 'primary.light', borderRadius: 2 }}>
          <Typography variant="h5" gutterBottom>
            Predicted House Price
          </Typography>
          {predicting ? (
            <CircularProgress size={24} />
          ) : prediction !== null ? (
            <Typography variant="h3">
              ${(prediction * 1000).toLocaleString()}
            </Typography>
          ) : (
            <Typography variant="body1" color="text.secondary">
              Adjust sliders to see prediction
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}
```

### 6. Main App Component

Create `src/App.jsx`:

```javascript
import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import { DemoPredictor } from './components/DemoPredictor';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
  },
});

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <Container>
          <h1>Something went wrong.</h1>
          <pre>{this.state.error?.message}</pre>
        </Container>
      );
    }

    return this.props.children;
  }
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <Container>
          <DemoPredictor />
        </Container>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App;
```

## Integration Steps

### Step-by-Step Implementation

1. **Setup Project Structure**
   ```
   your-project/
   ├── model-dev/
   │   ├── pyproject.toml
   │   └── build_model.py
   └── web/
       ├── package.json
       ├── vite.config.js
       ├── index.html
       └── src/
           ├── main.jsx
           ├── App.jsx
           ├── components/
           │   └── DemoPredictor.jsx
           ├── hooks/
           │   ├── useModelLoader.js
           │   └── usePredict.js
           └── workers/
               └── predict.worker.js
   ```

2. **Train and Export Model**
   ```bash
   cd model-dev
   pip install uv  # or use pip/conda
   uv sync
   uv run python build_model.py
   ```

3. **Install Web Dependencies**
   ```bash
   cd ../web
   npm install
   ```

4. **Run Development Server**
   ```bash
   npm run dev
   ```

### Adapting to Your Model

To use a different model, modify these parts:

1. **Model Training Script**
   ```python
   # Replace with your model
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

2. **Update Metadata Generation**
   ```python
   metadata = {
       "model": {
           "name": "Your Model Name",
           "type": "random_forest",
           "task": "classification"  # or "regression"
       },
       # Update based on your features
   }
   ```

3. **Adjust UI Component**
   - Update feature sliders based on your model's inputs
   - Modify prediction display for classification tasks

## Performance Optimization

### 1. Model Optimization

```python
# Reduce model size
from sklearn.tree import DecisionTreeRegressor

# Use simpler models for web deployment
model = DecisionTreeRegressor(max_depth=5)

# Or use feature selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=5)
X_reduced = selector.fit_transform(X, y)
```

### 2. Loading Performance

```javascript
// Preload model during app initialization
const modelPreloader = new Worker(
  new URL('./workers/predict.worker.js', import.meta.url),
  { type: 'module' }
);

// Cache model in IndexedDB for offline use
async function cacheModel() {
  const response = await fetch('/model.onnx');
  const blob = await response.blob();
  
  // Store in IndexedDB
  const db = await openDB('ml-models', 1);
  await db.put('models', blob, 'cached-model');
}
```

### 3. Inference Optimization

```javascript
// Batch predictions
async function batchPredict(featuresList) {
  const batchSize = featuresList.length;
  const inputData = new Float32Array(batchSize * numFeatures);
  
  // Fill batch tensor
  featuresList.forEach((features, i) => {
    inputData.set(features, i * numFeatures);
  });
  
  const inputTensor = new ort.Tensor(
    'float32', 
    inputData, 
    [batchSize, numFeatures]
  );
  
  return session.run({ [inputName]: inputTensor });
}
```

## Troubleshooting

### Common Issues and Solutions

1. **CORS Errors**
   ```javascript
   // vite.config.js
   server: {
     headers: {
       'Cross-Origin-Embedder-Policy': 'require-corp',
       'Cross-Origin-Opener-Policy': 'same-origin',
     }
   }
   ```

2. **WebAssembly Not Loading**
   ```javascript
   // Explicitly set WASM paths
   ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
   ```

3. **Model Too Large**
   ```python
   # Use quantization
   from onnxruntime.quantization import quantize_dynamic, QuantType
   
   quantize_dynamic(
       model_input='model.onnx',
       model_output='model_quant.onnx',
       weight_type=QuantType.QInt8
   )
   ```

4. **Memory Issues**
   ```javascript
   // Dispose tensors after use
   const result = await session.run(feeds);
   inputTensor.dispose();
   ```

### Browser Compatibility

```javascript
// Check for WebAssembly support
if (typeof WebAssembly === 'undefined') {
  throw new Error('WebAssembly not supported');
}

// Feature detection
const supportsWorkers = typeof Worker !== 'undefined';
const supportsWASM = (() => {
  try {
    if (typeof WebAssembly === 'object' &&
        typeof WebAssembly.instantiate === 'function') {
      const module = new WebAssembly.Module(
        Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00)
      );
      return module instanceof WebAssembly.Module;
    }
  } catch (e) {}
  return false;
})();
```

## Deployment

### 1. Build for Production

```bash
cd web
npm run build
```

### 2. Optimize Assets

```javascript
// vite.config.js
build: {
  rollupOptions: {
    output: {
      manualChunks: {
        'onnx-runtime': ['onnxruntime-web'],
        'mui': ['@mui/material'],
      }
    }
  },
  // Enable compression
  minify: 'terser',
  terserOptions: {
    compress: {
      drop_console: true,
    }
  }
}
```

### 3. Deploy to Static Hosting

The built app can be deployed to any static hosting service:

**Netlify**
```toml
# netlify.toml
[build]
  publish = "web/dist"

[[headers]]
  for = "/*"
  [headers.values]
    Cross-Origin-Embedder-Policy = "require-corp"
    Cross-Origin-Opener-Policy = "same-origin"
```

**Vercel**
```json
// vercel.json
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "Cross-Origin-Embedder-Policy",
          "value": "require-corp"
        },
        {
          "key": "Cross-Origin-Opener-Policy",
          "value": "same-origin"
        }
      ]
    }
  ]
}
```

### 4. CDN Configuration

For better performance, serve ONNX Runtime from CDN:

```html
<!-- index.html -->
<script type="module">
  // Preload ONNX Runtime
  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'script';
  link.href = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/ort.min.js';
  document.head.appendChild(link);
</script>
```

## Best Practices

1. **Model Selection**
   - Use lightweight models (< 100KB ideal)
   - Prefer tree-based models over deep neural networks
   - Consider model compression techniques

2. **Error Handling**
   - Implement comprehensive error boundaries
   - Provide fallback UI for unsupported browsers
   - Log errors for monitoring

3. **Performance Monitoring**
   ```javascript
   // Track inference time
   const startTime = performance.now();
   const result = await session.run(feeds);
   const inferenceTime = performance.now() - startTime;
   
   // Send to analytics
   analytics.track('ml_inference', {
     model: 'xgboost',
     time_ms: inferenceTime,
     input_size: features.length
   });
   ```

4. **Security Considerations**
   - Validate all inputs before inference
   - Implement rate limiting for predictions
   - Consider model IP protection needs

## Conclusion

This implementation provides a complete framework for running ML models in browsers. Key advantages:

- **No server costs** for inference
- **Instant predictions** without network latency
- **Privacy-preserving** as data stays on device
- **Offline capability** once model is loaded
- **Scalable** to millions of users

The architecture is flexible and can be adapted for various ML models and use cases, from simple linear regression to complex neural networks, as long as they can be exported to ONNX format.

For production deployments, consider:
- Model versioning and updates
- A/B testing different models
- Performance monitoring
- Progressive enhancement for older browsers
- Caching strategies for offline use

This approach represents the future of privacy-preserving, scalable ML applications on the web.