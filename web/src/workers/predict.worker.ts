/* eslint-disable no-restricted-globals */
/// <reference lib="webworker" />
declare const self: DedicatedWorkerGlobalScope;

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
        console.log('Loading model and metadata...');
        meta = data.meta;
        
        // Create inference session with WebAssembly backend (WebGPU may not be available)
        session = await ort.InferenceSession.create(`/${meta.model}`, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        
        console.log('Model loaded successfully');
        const response: WorkerOutgoingMessage = { event: 'loaded', success: true };
        postMessage(response);
      } catch (error) {
        console.error('Failed to load model:', error);
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
        const errorResponse: WorkerOutgoingMessage = { 
          event: 'prediction', 
          success: false, 
          error: 'Model not loaded' 
        };
        postMessage(errorResponse);
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
        const feeds: Record<string, ort.Tensor> = { [meta.input_names[0]]: inputTensor };
        const results = await session.run(feeds);
        
        // Extract prediction value
        const outputTensor = results[meta.output_name];
        const prediction = outputTensor.data[0] as number;
        
        const response: WorkerOutgoingMessage = { 
          event: 'prediction', 
          success: true, 
          value: prediction 
        };
        postMessage(response);
      } catch (error) {
        console.error('Prediction failed:', error);
        const errorResponse: WorkerOutgoingMessage = { 
          event: 'prediction', 
          success: false, 
          error: error instanceof Error ? error.message : 'Unknown error' 
        };
        postMessage(errorResponse);
      }
      break;

    default:
      // TypeScript exhaustiveness check
      const _exhaustiveCheck: never = data;
      console.warn('Unknown event:', (_exhaustiveCheck as any).event);
  }
};