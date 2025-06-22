import { useEffect, useState, useRef } from 'react';
import PredictWorker from '../workers/predict.worker.js?worker';

export function useModelLoader() {
  const [meta, setMeta] = useState(null);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const workerRef = useRef(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        // Fetch model metadata
        const response = await fetch('/model-package.json');
        if (!response.ok) {
          throw new Error('Failed to fetch model metadata');
        }
        const metadata = await response.json();
        setMeta(metadata);

        // Create and initialize worker
        const worker = new PredictWorker();
        workerRef.current = worker;

        // Set up message handler
        worker.onmessage = ({ data }) => {
          if (data.event === 'loaded') {
            if (data.success) {
              setReady(true);
              setLoading(false);
            } else {
              setError(data.error || 'Failed to load model');
              setLoading(false);
            }
          }
        };

        // Load model in worker
        worker.postMessage({ event: 'load', meta: metadata });

      } catch (err) {
        console.error('Model loading error:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    loadModel();

    // Cleanup
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);

  return { 
    meta, 
    ready, 
    loading, 
    error, 
    worker: workerRef.current 
  };
}