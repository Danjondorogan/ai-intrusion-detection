import React, { useRef, useEffect } from 'react';
import { LogEntry } from '../types';
import { AlertTriangle, Activity, Clock } from 'lucide-react';

interface Props {
  logs: LogEntry[];
}

export const LogViewer: React.FC<Props> = ({ logs }) => {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="bg-zinc-950 rounded-sm border border-zinc-900 flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-zinc-900 bg-zinc-950">
        <h3 className="text-zinc-500 font-bold text-[10px] uppercase tracking-widest">System Events</h3>
      </div>
      <div className="flex-grow overflow-y-auto p-0 space-y-0">
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-700 gap-2">
            <Clock className="w-6 h-6 opacity-20" />
            <span className="text-xs font-mono uppercase">Awaiting Data stream...</span>
          </div>
        ) : (
          logs.map((log) => (
            <div 
              key={log.id} 
              className={`px-4 py-3 border-b border-zinc-900 flex items-center justify-between text-xs font-mono ${
                log.prediction === 1 
                  ? 'bg-accent-900/10 text-accent-500' 
                  : 'bg-black text-zinc-400 hover:text-zinc-200'
              }`}
            >
              <div className="flex items-center gap-4">
                <span className="opacity-50 text-[10px]">
                  {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : '--:--:--'}
                </span>
                
                {log.status === 'warming_up' ? (
                   <span className="text-zinc-600 flex items-center gap-2 uppercase tracking-tight">
                      WARMUP
                   </span>
                ) : log.prediction === 1 ? (
                   <span className="text-accent-500 font-bold flex items-center gap-2 uppercase tracking-tight">
                      <AlertTriangle className="w-3 h-3" /> DOS ATTACK
                   </span>
                ) : (
                   <span className="text-zinc-500 flex items-center gap-2 uppercase tracking-tight">
                      <Activity className="w-3 h-3" /> NORMAL
                   </span>
                )}
              </div>
              
              <div className="flex items-center gap-4">
                 {log.dos_probability !== null && (
                    <span className={`font-bold ${
                           (log.dos_probability || 0) > 0.5 ? 'text-accent-500' : 'text-zinc-600'
                       }`}>
                          {((log.dos_probability || 0) * 100).toFixed(1)}%
                    </span>
                 )}
              </div>
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
};