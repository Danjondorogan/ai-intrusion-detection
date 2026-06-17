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
  const chartData = data.slice(-30).map((entry, index) => ({
    name: index,
    prob:
      entry.dos_probability !== null
        ? Number((entry.dos_probability * 100).toFixed(1))
        : 0,
    status: entry.status,
    isAttack: entry.prediction === 1
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const probability = payload[0].value;

      let color = 'text-white';
      let border = 'border-zinc-700';

      if (probability >= 50) {
        color = 'text-red-500';
        border = 'border-red-500';
      } else if (probability >= 30) {
        color = 'text-orange-500';
        border = 'border-orange-500';
      } else if (probability >= 10) {
        color = 'text-yellow-500';
        border = 'border-yellow-500';
      }

      return (
        <div className={`bg-black border p-2 rounded-sm shadow-xl ${border}`}>
          <p className="text-zinc-400 text-[10px] font-mono uppercase">
            Probability
          </p>
          <p className={`font-mono font-bold ${color}`}>
            {probability}%
          </p>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="bg-zinc-950 p-4 rounded-sm border border-zinc-900 h-64 w-full">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest">
          Live Analysis
        </h3>

        <div className="flex gap-4 text-[10px] font-mono uppercase text-zinc-600">
          <div className="flex items-center gap-1">
            <div className="w-2 h-0.5 bg-white"></div>
            Probability
          </div>

          <div className="flex items-center gap-1">
            <div className="w-2 h-0.5 bg-yellow-500"></div>
            Suspicious
          </div>

          <div className="flex items-center gap-1">
            <div className="w-2 h-0.5 bg-orange-500"></div>
            Threat
          </div>

          <div className="flex items-center gap-1">
            <div className="w-2 h-0.5 bg-red-500"></div>
            Attack
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height="80%">
        <LineChart data={chartData}>
          <CartesianGrid
            strokeDasharray="1 4"
            stroke="#27272a"
            vertical={false}
          />

          <XAxis dataKey="name" hide />

          <YAxis
            domain={[0, 100]}
            stroke="#52525b"
            fontSize={10}
            fontFamily="JetBrains Mono"
            tickFormatter={(val) => `${val}%`}
          />

          <Tooltip
            content={<CustomTooltip />}
            cursor={{
              stroke: '#3f3f46',
              strokeWidth: 1
            }}
          />

          {/* Suspicious */}
          <ReferenceLine
            y={10}
            stroke="#eab308"
            strokeDasharray="4 4"
            strokeOpacity={0.8}
          />

          {/* Threat */}
          <ReferenceLine
            y={30}
            stroke="#f97316"
            strokeDasharray="4 4"
            strokeOpacity={0.8}
          />

          {/* Attack */}
          <ReferenceLine
            y={50}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeOpacity={0.8}
          />

          <Line
            type="monotone"
            dataKey="prob"
            stroke="#ffffff"
            strokeWidth={2}
            activeDot={{
              r: 5,
              fill: '#ffffff'
            }}
            isAnimationActive={false}
            dot={(props: any) => {
              const value = props.payload.prob;

              let fill = '#ffffff';

              if (value >= 50) {
                fill = '#ef4444';
              } else if (value >= 30) {
                fill = '#f97316';
              } else if (value >= 10) {
                fill = '#eab308';
              }

              return (
                <circle
                  key={props.key}
                  cx={props.cx}
                  cy={props.cy}
                  r={3}
                  fill={fill}
                  stroke="none"
                />
              );
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};