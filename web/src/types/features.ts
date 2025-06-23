// Feature and prediction related types

export interface FeatureRange {
  min: number;
  max: number;
  default: number;
  step: number;
}

export interface PredictionResult {
  value: number;
  timestamp: number;
}

// Re-export FeatureData for convenience
export type { FeatureData } from './worker-messages';