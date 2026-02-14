export interface PredictionRequest {
  session_id: string;
  features: number[];
}

export interface PredictionResponse {
  status: 'warming_up' | 'ready';
  dos_probability: number | null;
  prediction: 0 | 1 | null;
  timesteps_collected: number | null;
  required_timesteps: number | null;
  timestamp?: number; // Added for frontend tracking
}

export interface LogEntry extends PredictionResponse {
  id: string;
}

export interface HealthStatus {
  status: 'online' | 'offline' | 'checking';
}

export const FEATURE_COUNT = 84;
