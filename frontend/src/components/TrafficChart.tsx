import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { LogEntry } from '../types';

interface Props {
  data: LogEntry[];
}

export const TrafficChart: React.FC<Props> = ({ data }) => {
  // Format data for chart
  const chartData = data.slice(-30).map((entry, index) => ({
    name: index,
    prob: entry.dos_probability !== null ? (entry.dos_probability * 100).toFixed(1) : 0,
    status: entry.status,
    isAttack: entry.prediction === 1
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const isAttack = payload[0].payload.isAttack;
      return (
        <div className={`border p-2 rounded-sm shadow-xl ${isAttack ? 'bg-accent-900 border-accent-500' : 'bg-black border-zinc-700'}`}>
          <p className="text-zinc-400 text-[10px] font-mono uppercase">Probability</p>
          <p className={`font-mono font-bold ${isAttack ? 'text-accent-500' : 'text-white'}`}>{payload[0].value}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-zinc-950 p-4 rounded-sm border border-zinc-900 h-64 w-full">
      <div className="flex justify-between items-center mb-6">
         <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Live Analysis</h3>
         <div className="flex gap-4 text-[10px] font-mono uppercase text-zinc-600">
            <div className="flex items-center gap-1"><div className="w-2 h-0.5 bg-white"></div>Probability</div>
            <div className="flex items-center gap-1"><div className="w-2 h-0.5 bg-accent-500"></div>Threat</div>
         </div>
      </div>
      <ResponsiveContainer width="100%" height="80%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="1 4" stroke="#27272a" vertical={false} />
          <XAxis dataKey="name" hide />
          <YAxis domain={[0, 100]} stroke="#52525b" fontSize={10} fontFamily="JetBrains Mono" tickFormatter={(val) => `${val}%`} />
          <Tooltip content={<CustomTooltip />} cursor={{stroke: '#3f3f46', strokeWidth: 1}} />
          <ReferenceLine y={50} stroke="#dc2626" strokeDasharray="2 2" strokeOpacity={0.5} />
          <Line 
            type="stepAfter" 
            dataKey="prob" 
            stroke="#ffffff" 
            strokeWidth={1.5} 
            dot={(props: any) => {
                // Custom dot logic: Red dot for attack, invisible for normal
                if (props.payload.isAttack) {
                    return <circle cx={props.cx} cy={props.cy} r={3} fill="#ef4444" stroke="none" key={props.key} />
                }
                return null;
            }}
            activeDot={{ r: 4, fill: '#fff' }} 
            isAnimationActive={false} 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};