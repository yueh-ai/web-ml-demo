import { useEffect, useState, useRef } from 'react';
import PredictWorker from '../workers/predict.worker.ts?worker';
import type { ModelMetadata, WorkerOutgoingMessage } from '../types';

interface UseModelLoaderReturn {
  meta: ModelMetadata | null;
  ready: boolean;
  loading: boolean;
  error: string | null;
  worker: Worker | null;
}

export function useModelLoader(): UseModelLoaderReturn {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [ready, setReady] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);

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
        worker.onmessage = ({ data }: MessageEvent<WorkerOutgoingMessage>) => {
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
        setError(err instanceof Error ? err.message : 'Failed to load model');
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