import React from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';

interface Props {
  probability: number | null; // 0 to 1
  status: 'warming_up' | 'ready';
  prediction: 0 | 1 | null;
}

export const ProbabilityGauge: React.FC<Props> = ({ probability, status, prediction }) => {
  const isAttack = prediction === 1;
  const isSafe = prediction === 0;
  const percentage = probability ? probability * 100 : 0;
  
  // Minimalist Color logic
  let strokeColor = '#27272a'; // Zinc 800 (Empty/Warmup)
  
  if (status === 'ready') {
    if (isAttack) {
        strokeColor = '#ef4444'; // Red
    } else {
        strokeColor = '#ffffff'; // White
    }
  }

  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative flex flex-col items-center justify-center p-6 bg-zinc-950 rounded-sm border border-zinc-900 h-full">
      <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-6">Threat Probability</h3>
      
      <div className="relative w-48 h-48">
        {/* Background Circle */}
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
          {/* Progress Circle */}
          <circle
            cx="96"
            cy="96"
            r={radius}
            stroke={strokeColor}
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="butt" 
            className="transition-all duration-700 ease-out"
          />
        </svg>
        
        {/* Center Content */}
        <div className="absolute top-0 left-0 w-full h-full flex flex-col items-center justify-center">
            {status === 'warming_up' ? (
                 <span className="text-4xl font-mono font-light text-zinc-700">--</span>
            ) : (
                <>
                <span className={`text-5xl font-mono font-light tracking-tighter ${isAttack ? 'text-accent-500' : 'text-white'}`}>
                    {percentage.toFixed(0)}<span className="text-2xl text-zinc-600">%</span>
                </span>
                <span className={`text-[10px] font-bold mt-2 uppercase tracking-widest px-2 py-0.5 border ${
                    isAttack ? 'border-accent-500 text-accent-500' : isSafe ? 'border-zinc-700 text-zinc-500' : 'text-zinc-500 border-zinc-800'
                }`}>
                    {isAttack ? 'CRITICAL' : isSafe ? 'NORMAL' : 'ANALYZING'}
                </span>
                </>
            )}
        </div>
      </div>

      <div className="mt-8 flex items-center gap-3">
        {isAttack ? (
             <ShieldAlert className="w-5 h-5 text-accent-500 animate-pulse" />
        ) : (
             <ShieldCheck className="w-5 h-5 text-zinc-600" />
        )}
        <span className={`text-sm font-medium tracking-widest uppercase ${
            isAttack ? 'text-accent-500' : isSafe ? 'text-white' : 'text-zinc-600'
        }`}>
            {status === 'warming_up' ? 'BUFFERING...' : (isAttack ? 'DoS DETECTED' : 'TRAFFIC NORMAL')}
        </span>
      </div>
    </div>
  );
};