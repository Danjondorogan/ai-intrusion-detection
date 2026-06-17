export interface PredictionRequest {
  session_id: string;
  features: number[];
}

export interface PredictionResponse {
  status: 'warming_up' | 'monitoring' | 'attack';

  dos_probability: number | null;

  prediction: number | null;

  timesteps_collected: number | null;

  required_timesteps: number | null;

  consecutive_detections?: number;

  latency_ms?: number;

  timestamp?: number;
  top_features?: {
  name: string;
  impact: number;
}[];

}

export interface LogEntry extends PredictionResponse {
  id: string;
}

export interface HealthStatus {
  status: string;
}
export const FEATURE_COUNT = 840;
