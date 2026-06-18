import React from "react";
import { Cpu } from "lucide-react";

export const ModelInfoCard: React.FC = () => {
  return (
    <div className="bg-zinc-950 border border-zinc-900 rounded-sm p-5">

      <div className="flex items-center gap-2 mb-5">
        <Cpu className="w-4 h-4 text-cyan-400" />
        <h3 className="text-xs uppercase tracking-widest font-bold text-zinc-400">
          Model Information
        </h3>
      </div>

      <div className="space-y-3 text-sm">

        <div className="flex justify-between border-b border-zinc-900 pb-2">
          <span className="text-zinc-500">Model Type</span>
          <span className="text-white font-mono">LSTM</span>
        </div>

        <div className="flex justify-between border-b border-zinc-900 pb-2">
          <span className="text-zinc-500">Window Size</span>
          <span className="text-white font-mono">10 Timesteps</span>
        </div>

        <div className="flex justify-between border-b border-zinc-900 pb-2">
          <span className="text-zinc-500">Features / Timestep</span>
          <span className="text-white font-mono">84</span>
        </div>

        <div className="flex justify-between border-b border-zinc-900 pb-2">
          <span className="text-zinc-500">Input Vector</span>
          <span className="text-white font-mono">840 Features</span>
        </div>

        <div className="flex justify-between border-b border-zinc-900 pb-2">
          <span className="text-zinc-500">Framework</span>
          <span className="text-white font-mono">TensorFlow</span>
        </div>

        <div className="flex justify-between">
          <span className="text-zinc-500">Detection Mode</span>
          <span className="text-emerald-400 font-mono">
            Real-Time
          </span>
        </div>

      </div>

    </div>
  );
};