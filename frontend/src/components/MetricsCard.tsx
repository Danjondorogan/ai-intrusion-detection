import React from 'react';
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  Cpu
} from 'lucide-react';

interface Props {
  status: string;
  collected: number | null;
  required: number | null;
}

export const MetricsCard: React.FC<Props> = ({
  status,
  collected,
  required
}) => {

  const current =
    status === "warming_up"
      ? (collected || 0)
      : 10;

  const total = required || 10;

  const progress = Math.min(
    (current / total) * 100,
    100
  );

  const isReady = status !== "warming_up";

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

      {/* SYSTEM STATE */}

      <div className="bg-zinc-950 p-6 rounded-sm border border-zinc-900 flex items-center justify-between">

        <div>

          <p className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-2">
            System State
          </p>

          <h3
            className={`text-xl font-medium flex items-center gap-3 ${
              status === "attack"
                ? "text-red-500"
                : isReady
                ? "text-white"
                : "text-zinc-400"
            }`}
          >

            <div
              className={`p-1.5 rounded-sm ${
                status === "attack"
                  ? "bg-red-500 text-black"
                  : isReady
                  ? "bg-white text-black"
                  : "bg-zinc-800 text-zinc-500"
              }`}
            >
              {status === "attack" ? (
                <ShieldAlert className="w-4 h-4" />
              ) : (
                <Shield className="w-4 h-4" />
              )}
            </div>

            {status === "warming_up"
              ? "INITIALIZING"
              : status === "attack"
              ? "ATTACK DETECTED"
              : "ACTIVE"}

          </h3>

        </div>

      </div>

      {/* DETECTION ENGINE */}

      <div className="bg-zinc-950 p-6 rounded-sm border border-zinc-900 flex flex-col justify-center">

        <div className="flex justify-between items-center mb-4">

          <div>

            <p className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-2 flex items-center gap-2">
              <Cpu className="w-3 h-3" />
              Detection Engine
            </p>

            <div className="text-xl font-mono text-white">
              LSTM ACTIVE
            </div>

          </div>

          <div>

            {status === "attack" ? (
              <ShieldAlert className="w-6 h-6 text-red-500" />
            ) : (
              <ShieldCheck className="w-6 h-6 text-white" />
            )}

          </div>

        </div>

        <div className="w-full bg-zinc-900 h-1 rounded-none overflow-hidden">

          <div
            className="h-full bg-white transition-all duration-700"
            style={{
              width: `${progress}%`
            }}
          />

        </div>

        <div className="flex justify-between mt-3 text-[10px] uppercase tracking-widest">

          <span className="text-zinc-600">
            Real-Time Analysis
          </span>

          <span
            className={
              status === "attack"
                ? "text-red-500"
                : "text-zinc-400"
            }
          >
            {status === "attack"
              ? "Threat Active"
              : "Monitoring"}
          </span>

        </div>

      </div>

    </div>
  );
};