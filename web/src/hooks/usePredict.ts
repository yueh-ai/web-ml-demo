import { useEffect, useState, useRef } from 'react';
import type { ModelMetadata, FeatureData, WorkerOutgoingMessage } from '../types';

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