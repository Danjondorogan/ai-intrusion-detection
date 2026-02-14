import React, { useState, useEffect, useCallback } from "react";

import { Header } from "./components/Header";
import { ProbabilityGauge } from "./components/ProbabilityGauge";
import { ControlPanel } from "./components/ControlPanel";
import { LogViewer } from "./components/LogViewer";
import { MetricsCard } from "./components/MetricsCard";
import { TrafficChart } from "./components/TrafficChart";

import { checkHealth, sendPrediction } from "./services/api";
import { LogEntry, PredictionResponse, FEATURE_COUNT } from "./types";

/* Add small variation for auto mode */
const jitterFeatures = (base: number[]): number[] => {
  return base.map((f) => {
    const noise = (Math.random() - 0.5) * 0.1;
    return Number((f + noise).toFixed(6));
  });
};

function App() {
  const [isOnline, setIsOnline] = useState(false);
  const [sessionId, setSessionId] = useState("session_alpha");
  const [featuresRaw, setFeaturesRaw] = useState("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentStatus, setCurrentStatus] =
    useState<PredictionResponse | null>(null);
  const [isAutoSending, setIsAutoSending] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  /* -------------------------------------------------- */
  /* Backend Health Check */
  /* -------------------------------------------------- */
  useEffect(() => {
    const check = async () => {
      const ok = await checkHealth();
      setIsOnline(ok);
    };

    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, []);

  /* -------------------------------------------------- */
  /* Send Prediction */
  /* -------------------------------------------------- */
  const handleSend = useCallback(async () => {
    if (!sessionId) return;

    const parsed = featuresRaw
      .split(/[\s,]+/)
      .filter((v) => v.trim() !== "")
      .map(Number);

    if (parsed.length !== FEATURE_COUNT) {
      alert(`Please enter exactly ${FEATURE_COUNT} features`);
      return;
    }

    setIsLoading(true);

    try {
      const response = await sendPrediction({
        session_id: sessionId,
        features: parsed,
      });

      const newLog: LogEntry = {
        ...response,
        id: crypto.randomUUID(),
      };

      setCurrentStatus(response);
      setLogs((prev) => [...prev, newLog].slice(-100));

      if (isAutoSending) {
        const next = jitterFeatures(parsed);
        setFeaturesRaw(next.join(", "));
      }
    } catch (err) {
      console.error("Backend error:", err);
      setIsAutoSending(false);
      setIsOnline(false);
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, featuresRaw, isAutoSending]);

  /* -------------------------------------------------- */
  /* Auto Mode */
  /* -------------------------------------------------- */
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | undefined;

    if (isAutoSending) {
      interval = setInterval(handleSend, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAutoSending, handleSend]);

  /* -------------------------------------------------- */
  /* UI */
  /* -------------------------------------------------- */
  return (
    <div className="min-h-screen bg-black text-zinc-100 font-sans">
      <Header isOnline={isOnline} />

      <main className="max-w-7xl mx-auto p-4 md:p-6 lg:p-8 space-y-6">

        {/* Project Description (Research Grade Section) */}
        <div className="bg-zinc-900 rounded-xl p-5 border border-zinc-800">
          <h2 className="text-xl font-semibold mb-2">
            AI-Based Intrusion Detection System
          </h2>
          <p className="text-sm text-zinc-400 leading-relaxed">
            This system uses a deep learning LSTM model to detect Distributed
            Denial-of-Service (DoS) attacks from network traffic patterns.
            Temporal flow features are analyzed in real time to estimate attack
            probability and classify traffic as normal or malicious.
          </p>
        </div>

        {/* Top Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div>
            <ProbabilityGauge
              probability={currentStatus?.dos_probability ?? null}
              status={currentStatus?.status ?? "warming_up"}
              prediction={currentStatus?.prediction ?? null}
            />
          </div>

          <div className="lg:col-span-2 flex flex-col gap-6">
            <MetricsCard
              status={currentStatus?.status ?? "warming_up"}
              collected={currentStatus?.timesteps_collected ?? null}
              required={currentStatus?.required_timesteps ?? null}
            />
            <TrafficChart data={logs} />
          </div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[500px]">
          <div className="h-full">
            <ControlPanel
              sessionId={sessionId}
              setSessionId={setSessionId}
              featuresRaw={featuresRaw}
              setFeaturesRaw={setFeaturesRaw}
              onSend={handleSend}
              isAutoSending={isAutoSending}
              setIsAutoSending={setIsAutoSending}
              isLoading={isLoading}
            />
          </div>

          <div className="lg:col-span-2 h-full">
            <LogViewer logs={logs} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;