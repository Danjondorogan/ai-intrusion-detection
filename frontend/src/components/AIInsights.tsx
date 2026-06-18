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

  const isAttack = threat >= 50;
  const isSuspicious = threat >= 15 && threat < 50;

  const accentColor = isAttack
    ? "text-red-400"
    : isSuspicious
    ? "text-yellow-400"
    : "text-emerald-400";

  const accentBar = isAttack
    ? "bg-red-500"
    : isSuspicious
    ? "bg-yellow-500"
    : "bg-emerald-500";

  const accentBorder = isAttack
    ? "border-red-500/30"
    : isSuspicious
    ? "border-yellow-500/30"
    : "border-emerald-500/30";

  const statusLabel = isAttack
    ? "ATTACK DETECTED"
    : isSuspicious
    ? "SUSPICIOUS TRAFFIC"
    : "NORMAL TRAFFIC";

  return (
    <div className="bg-zinc-950 border border-zinc-900 rounded-sm p-5">

      {/* Header */}
      <div className="flex items-center gap-2 mb-5">
        <Brain className={`w-4 h-4 ${accentColor}`} />
        <h3 className="text-xs uppercase tracking-widest font-bold text-zinc-400">
          AI Insights
        </h3>
      </div>

      {/* Top Cards */}
      <div className="grid grid-cols-2 gap-4 mb-5">

        <div className={`bg-black border p-3 rounded-sm ${accentBorder}`}>
          <p className="text-zinc-500 text-[10px] uppercase tracking-widest">
            Threat Score
          </p>

          <div className={`text-2xl font-mono mt-1 ${accentColor}`}>
            {threat.toFixed(1)}%
          </div>

          <div className={`text-[10px] mt-1 font-bold uppercase tracking-widest ${accentColor}`}>
            {statusLabel}
          </div>
        </div>

        <div className="bg-black border border-zinc-800 p-3 rounded-sm">
          <p className="text-zinc-500 text-[10px] uppercase tracking-widest">
            Inference Time
          </p>

          <div className="text-2xl font-mono text-white mt-1">
            {(latency ?? 0).toFixed(2)} ms
          </div>
        </div>

      </div>

      {/* Features Header */}
      <div className="mb-4 flex items-center gap-2">
        <ShieldAlert className={`w-4 h-4 ${accentColor}`} />

        <span className="text-xs uppercase tracking-widest text-zinc-400">
          Top Influencing Features
        </span>
      </div>

      {/* Features */}
      <div className="space-y-3">

        {(features ?? []).map((feature, index) => (
          <div key={index}>

            <div className="flex justify-between text-xs mb-1">
              <span className="text-zinc-300">
                {feature.name}
              </span>

              <span className={accentColor}>
                {feature.impact}%
              </span>
            </div>

            <div className="w-full bg-zinc-900 h-2 rounded-sm overflow-hidden">

              <div
                className={`h-full ${accentBar}`}
                style={{
                  width: `${feature.impact}%`,
                }}
              />

            </div>

          </div>
        ))}

      </div>
        {/* Explanation */}
        <div className="mt-6 border-t border-zinc-900 pt-4">

        <p className="text-zinc-500 text-[10px] uppercase tracking-widest mb-2">
            AI Assessment
        </p>

        <p className="text-xs text-zinc-400 leading-relaxed">

            {threat >= 50
            ? "The LSTM model identified temporal traffic behaviour strongly matching previously observed Denial-of-Service attack patterns. Elevated packet transmission rates and abnormal flow statistics contributed significantly to the final attack classification."

            : threat >= 15
            ? "The traffic exhibits moderate deviations from expected network behaviour. Although not currently classified as a confirmed attack, continued monitoring is recommended due to elevated anomaly indicators."

            : "Observed network traffic aligns with learned benign behaviour patterns. Flow statistics remain within expected operating ranges and no significant threat indicators were detected."
            }

        </p>

        </div>

    </div>
  );
};