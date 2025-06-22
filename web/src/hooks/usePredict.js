import { useEffect, useState, useRef } from 'react';

export function usePredict({ worker, meta, features }) {
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (!worker || !meta || !features) return;

    // Clear previous timeout
    clearTimeout(debounceRef.current);

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
      clearTimeout(debounceRef.current);
    };
  }, [features, worker, meta]);

  useEffect(() => {
    if (!worker) return;

    const handleMessage = ({ data }) => {
      if (data.event === 'prediction') {
        if (data.success) {
          setPrediction(data.value);
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