import React, { useRef, useEffect } from 'react';
import { LogEntry } from '../types';
import {
  AlertTriangle,
  Activity,
  Clock,
  ShieldAlert,
  ShieldCheck
} from 'lucide-react';

interface Props {
  logs: LogEntry[];
}

export const LogViewer: React.FC<Props> = ({ logs }) => {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getSeverity = (probability: number) => {
    const p = probability * 100;

    if (p >= 50)
      return {
        label: 'ATTACK',
        color: 'text-red-500',
        bg: 'bg-red-950/20'
      };

    if (p >= 30)
      return {
        label: 'THREAT',
        color: 'text-orange-500',
        bg: 'bg-orange-950/20'
      };

    if (p >= 10)
      return {
        label: 'SUSPICIOUS',
        color: 'text-yellow-500',
        bg: 'bg-yellow-950/10'
      };

    return {
      label: 'NORMAL',
      color: 'text-emerald-500',
      bg: 'bg-black'
    };
  };

  return (
    <div className="bg-zinc-950 rounded-sm border border-zinc-900 flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-zinc-900">
        <div className="flex items-center justify-between">
          <h3 className="text-zinc-500 font-bold text-[10px] uppercase tracking-widest">
            Security Events
          </h3>

          <span className="text-[10px] text-zinc-600 font-mono">
            {logs.length} EVENTS
          </span>
        </div>
      </div>

      <div className="flex-grow overflow-y-auto">
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-700 gap-2">
            <Clock className="w-6 h-6 opacity-20" />
            <span className="text-xs font-mono uppercase">
              Awaiting Data Stream...
            </span>
          </div>
        ) : (
          logs.map((log) => {
            const probability = log.dos_probability || 0;
            const severity = getSeverity(probability);

            return (
              <div
                key={log.id}
                className={`px-4 py-3 border-b border-zinc-900 flex items-center justify-between text-xs font-mono transition-colors ${severity.bg}`}
              >
                <div className="flex items-center gap-4">
                  <span className="opacity-50 text-[10px] min-w-[70px]">
                    {log.timestamp
                      ? new Date(log.timestamp).toLocaleTimeString()
                      : '--:--:--'}
                  </span>

                  {log.status === 'warming_up' ? (
                    <span className="text-zinc-500 flex items-center gap-2 uppercase">
                      <Clock className="w-3 h-3" />
                      BUFFERING
                    </span>
                  ) : probability >= 0.5 ? (
                    <span className="text-red-500 font-bold flex items-center gap-2 uppercase">
                      <AlertTriangle className="w-3 h-3" />
                      DOS ATTACK
                    </span>
                  ) : probability >= 0.3 ? (
                    <span className="text-orange-500 font-bold flex items-center gap-2 uppercase">
                      <ShieldAlert className="w-3 h-3" />
                      THREAT DETECTED
                    </span>
                  ) : probability >= 0.1 ? (
                    <span className="text-yellow-500 font-bold flex items-center gap-2 uppercase">
                      <Activity className="w-3 h-3" />
                      SUSPICIOUS TRAFFIC
                    </span>
                  ) : (
                    <span className="text-emerald-500 flex items-center gap-2 uppercase">
                      <ShieldCheck className="w-3 h-3" />
                      NORMAL TRAFFIC
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-4">
                  <span
                    className={`font-bold text-[11px] px-2 py-1 rounded-sm border ${severity.color}`}
                  >
                    {severity.label}
                  </span>

                  <span className={`font-bold text-sm ${severity.color}`}>
                    {(probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })
        )}

        <div ref={endRef} />
      </div>
    </div>
  );
};