import React from 'react';
import { Database, Zap } from 'lucide-react';

interface Props {
  status: 'warming_up' | 'ready';
  collected: number | null;
  required: number | null;
}

export const MetricsCard: React.FC<Props> = ({ status, collected, required }) => {
  const current = collected || 0;
  const total = required || 60; // Default fallback
  const progress = Math.min((current / total) * 100, 100);
  const isReady = status === 'ready';

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Model Status Card */}
        <div className="bg-zinc-950 p-6 rounded-sm border border-zinc-900 flex items-center justify-between">
            <div>
                <p className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-2">System State</p>
                <h3 className={`text-xl font-medium flex items-center gap-3 ${isReady ? 'text-white' : 'text-zinc-400'}`}>
                    <div className={`p-1.5 rounded-sm ${isReady ? 'bg-white text-black' : 'bg-zinc-800 text-zinc-500'}`}>
                        <Zap className="w-4 h-4" />
                    </div>
                    {status === 'warming_up' ? 'WARMING UP' : 'ACTIVE'}
                </h3>
            </div>
        </div>

        {/* Buffer Status Card */}
        <div className="bg-zinc-950 p-6 rounded-sm border border-zinc-900 flex flex-col justify-center">
             <div className="flex justify-between items-end mb-3">
                <div>
                    <p className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-2 flex items-center gap-2">
                        <Database className="w-3 h-3" /> Temporal Buffer
                    </p>
                    <div className="text-xl font-mono text-white">
                        {current} <span className="text-zinc-600 text-sm">/ {total}</span>
                    </div>
                </div>
                <div className="text-right">
                    <span className={`text-xl font-mono ${progress >= 100 ? 'text-white' : 'text-zinc-500'}`}>
                        {progress.toFixed(0)}%
                    </span>
                </div>
             </div>
             
             {/* Minimalist Progress Bar */}
             <div className="w-full bg-zinc-900 h-1 rounded-none overflow-hidden">
                <div 
                    className={`h-full transition-all duration-500 ease-out ${progress >= 100 ? 'bg-white' : 'bg-zinc-600'}`}
                    style={{ width: `${progress}%` }}
                />
             </div>
        </div>
    </div>
  );
};