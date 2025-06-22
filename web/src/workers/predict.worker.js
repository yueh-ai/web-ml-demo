/* eslint-disable no-restricted-globals */
import * as ort from 'onnxruntime-web';

let session = null;
let meta = null;

self.onmessage = async ({ data }) => {
  switch (data.event) {
    case 'load':
      try {
        console.log('Loading model and metadata...');
        meta = data.meta;
        
        // Create inference session with WebAssembly backend (WebGPU may not be available)
        session = await ort.InferenceSession.create(`/${meta.model}`, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        
        console.log('Model loaded successfully');
        postMessage({ event: 'loaded', success: true });
      } catch (error) {
        console.error('Failed to load model:', error);
        postMessage({ event: 'loaded', success: false, error: error.message });
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
        // Extract features in the correct order
        const features = meta.feature_order.map(name => data.features[name]);
        
        // Create input tensor
        const inputTensor = new ort.Tensor(
          'float32',
          Float32Array.from(features),
          [1, features.length]
        );
        
        // Run inference
        const feeds = { [meta.input_names[0]]: inputTensor };
        const results = await session.run(feeds);
        
        // Extract prediction value
        const outputTensor = results[meta.output_name];
        const prediction = outputTensor.data[0];
        
        postMessage({ 
          event: 'prediction', 
          success: true, 
          value: prediction 
        });
      } catch (error) {
        console.error('Prediction failed:', error);
        postMessage({ 
          event: 'prediction', 
          success: false, 
          error: error.message 
        });
      }
      break;

    default:
      console.warn('Unknown event:', data.event);
  }
};