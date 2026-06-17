import React from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';

interface Props {
  probability: number | null;
  status: string;
  prediction: number | null;
}

export const ProbabilityGauge: React.FC<Props> = ({
  probability,
  status,
  prediction
}) => {
  const percentage = probability ? probability * 100 : 0;

  let strokeColor = '#27272a';
  let textColor = 'text-zinc-500';
  let badgeColor = 'border-zinc-700 text-zinc-500';
  let label = 'NORMAL';
  let bottomText = 'TRAFFIC NORMAL';
  let isThreat = false;

  if (status !== 'warming_up') {
    if (percentage < 10) {
      strokeColor = '#ffffff';
      textColor = 'text-white';
      badgeColor = 'border-zinc-700 text-zinc-400';
      label = 'NORMAL';
      bottomText = 'TRAFFIC NORMAL';
    }
    else if (percentage < 30) {
      strokeColor = '#eab308';
      textColor = 'text-yellow-500';
      badgeColor = 'border-yellow-500 text-yellow-500';
      label = 'SUSPICIOUS';
      bottomText = 'SUSPICIOUS TRAFFIC';
      isThreat = true;
    }
    else if (percentage < 50) {
      strokeColor = '#f97316';
      textColor = 'text-orange-500';
      badgeColor = 'border-orange-500 text-orange-500';
      label = 'THREAT';
      bottomText = 'THREAT DETECTED';
      isThreat = true;
    }
    else {
      strokeColor = '#ef4444';
      textColor = 'text-red-500';
      badgeColor = 'border-red-500 text-red-500';
      label = 'DOS ATTACK';
      bottomText = 'ATTACK DETECTED';
      isThreat = true;
    }
  }

  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset =
    circumference - (percentage / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center justify-center p-6 bg-zinc-950 rounded-sm border border-zinc-900 h-full">
      <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-6">
        Threat Probability
      </h3>

      <div className="relative w-48 h-48">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="96"
            cy="96"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-zinc-900"
          />

          <circle
            cx="96"
            cy="96"
            r={radius}
            stroke={strokeColor}
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        <div className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center">
          {status === 'warming_up' ? (
            <>
              <span className="text-4xl font-mono font-light text-zinc-700">
                --
              </span>
              <span className="text-[10px] text-zinc-700 uppercase tracking-widest mt-2">
                BUFFERING
              </span>
            </>
          ) : (
            <>
              <span
                className={`text-5xl font-mono font-light tracking-tighter ${textColor}`}
              >
                {percentage.toFixed(0)}
                <span className="text-2xl text-zinc-600">%</span>
              </span>

              <span
                className={`text-[10px] font-bold mt-2 uppercase tracking-widest px-3 py-1 border ${badgeColor}`}
              >
                {label}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="mt-8 flex items-center gap-3">
        {isThreat ? (
          <ShieldAlert className="w-5 h-5 text-red-500 animate-pulse" />
        ) : (
          <ShieldCheck className="w-5 h-5 text-zinc-500" />
        )}

        <span
          className={`text-sm font-medium tracking-widest uppercase ${
            percentage < 10
              ? 'text-white'
              : percentage < 30
              ? 'text-yellow-500'
              : percentage < 50
              ? 'text-orange-500'
              : 'text-red-500'
          }`}
        >
          {status === 'warming_up'
            ? 'BUFFERING...'
            : bottomText}
        </span>
      </div>
    </div>
  );
};