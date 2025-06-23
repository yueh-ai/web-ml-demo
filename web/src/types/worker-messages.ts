// Worker message types for type-safe communication between main thread and worker

// Messages sent TO the worker
export interface LoadModelMessage {
  event: 'load';
  meta: ModelMetadata;
}

export interface PredictMessage {
  event: 'predict';
  features: FeatureData;
}

export type WorkerIncomingMessage = LoadModelMessage | PredictMessage;

// Messages sent FROM the worker
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

export type WorkerOutgoingMessage = LoadedMessage | PredictionMessage;

// Model metadata structure
export interface ModelMetadata {
  model: string;
  model_info: {
    type: string;
    r2_score: number;
    training_samples: number;
  };
  feature_order: string[];
  feature_ranges: {
    [featureName: string]: {
      min: number;
      max: number;
      default: number;
      step: number;
    };
  };
  input_names: string[];
  output_name: string;
}

// Feature data structure
export type FeatureData = {
  [featureName: string]: number;
};