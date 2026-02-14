// frontend/src/services/api.ts
import { PredictionRequest, PredictionResponse } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export const checkHealth = async (): Promise<boolean> => {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch (e) {
    return false;
  }
};

export const sendPrediction = async (data: PredictionRequest): Promise<PredictionResponse> => {
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!res.ok) {
      // surface backend error text for debugging
      const txt = await res.text();
      throw new Error(`API Error: ${res.status} ${res.statusText} - ${txt}`);
    }

    const result: PredictionResponse = await res.json();
    return { ...result, timestamp: Date.now() };
  } catch (error) {
    console.error("Prediction failed:", error);
    throw error;
  }
};