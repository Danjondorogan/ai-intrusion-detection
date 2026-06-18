import React from "react";
import { ShieldAlert, Activity, AlertTriangle, BarChart3 } from "lucide-react";

interface Props {
  logs: any[];
}

export const ThreatStats: React.FC<Props> = ({ logs }) => {

  const total = logs.length;

  const attacks = logs.filter(
    (l) => l.prediction === 1
  ).length;

  const suspicious = logs.filter(
    (l) =>
      l.dos_probability !== null &&
      l.dos_probability >= 0.15 &&
      l.dos_probability < 0.5
  ).length;

  const detectionRate =
    total > 0
      ? ((attacks / total) * 100).toFixed(1)
      : "0";

  const cards = [
    {
      title: "Analyses",
      value: total,
      icon: <BarChart3 className="w-4 h-4" />,
      color: "text-cyan-400"
    },
    {
      title: "Attacks",
      value: attacks,
      icon: <ShieldAlert className="w-4 h-4" />,
      color: "text-red-400"
    },
    {
      title: "Suspicious",
      value: suspicious,
      icon: <AlertTriangle className="w-4 h-4" />,
      color: "text-yellow-400"
    },
    {
      title: "Detection %",
      value: `${detectionRate}%`,
      icon: <Activity className="w-4 h-4" />,
      color: "text-emerald-400"
    }
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">

      {cards.map((card) => (
        <div
          key={card.title}
          className="bg-zinc-950 border border-zinc-900 rounded-sm p-4"
        >
          <div className={`mb-3 ${card.color}`}>
            {card.icon}
          </div>

          <div className="text-2xl font-mono text-white">
            {card.value}
          </div>

          <div className="text-[10px] uppercase tracking-widest text-zinc-500 mt-1">
            {card.title}
          </div>
        </div>
      ))}

    </div>
  );
};