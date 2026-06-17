import React from "react";
import { Brain, ShieldAlert } from "lucide-react";

interface Feature {
  name: string;
  impact: number;
}

interface Props {
  features?: Feature[];
  probability?: number | null;
  latency?: number | null;
}

export const AIInsights: React.FC<Props> = ({
  features,
  probability,
  latency,
}) => {
  const threat =
    probability !== undefined &&
    probability !== null
      ? probability * 100
      : 0;

  return (
    <div className="bg-zinc-950 border border-zinc-900 rounded-sm p-5">

      <div className="flex items-center gap-2 mb-5">
        <Brain className="w-4 h-4 text-blue-400" />
        <h3 className="text-xs uppercase tracking-widest font-bold text-zinc-400">
          AI Insights
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-5">

        <div className="bg-black border border-zinc-800 p-3 rounded-sm">
          <p className="text-zinc-500 text-[10px] uppercase tracking-widest">
            Threat Score
          </p>

          <div className="text-2xl font-mono text-white mt-1">
            {threat.toFixed(1)}%
          </div>
        </div>

        <div className="bg-black border border-zinc-800 p-3 rounded-sm">
          <p className="text-zinc-500 text-[10px] uppercase tracking-widest">
            Inference Time
          </p>

          <div className="text-2xl font-mono text-white mt-1">
            {latency ?? 0} ms
          </div>
        </div>

      </div>

      <div className="mb-4 flex items-center gap-2">
        <ShieldAlert className="w-4 h-4 text-red-400" />
        <span className="text-xs uppercase tracking-widest text-zinc-400">
          Top Influencing Features
        </span>
      </div>

      <div className="space-y-3">

        {(features ?? []).map((feature, index) => (
          <div key={index}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-zinc-300">
                {feature.name}
              </span>

              <span className="text-red-400">
                {feature.impact}%
              </span>
            </div>

            <div className="w-full bg-zinc-900 h-2 rounded-sm overflow-hidden">
              <div
                className="h-full bg-red-500"
                style={{
                  width: `${feature.impact}%`,
                }}
              />
            </div>
          </div>
        ))}

      </div>

      <div className="mt-6 border-t border-zinc-900 pt-4">

        <p className="text-zinc-500 text-[10px] uppercase tracking-widest mb-2">
          AI Explanation
        </p>

        <p className="text-xs text-zinc-400 leading-relaxed">
          The LSTM model analyzed temporal network traffic behaviour
          across multiple time windows and identified anomalous flow
          characteristics. Feature importance values represent the
          strongest contributors affecting the final attack probability.
        </p>

      </div>

    </div>
  );
};