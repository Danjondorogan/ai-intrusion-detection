import React from 'react';
import { Activity, Server } from 'lucide-react';

interface HeaderProps {
  isOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({ isOnline }) => {
  return (
    <header className="w-full bg-black border-b border-zinc-900 p-4 sticky top-0 z-50 backdrop-blur-md bg-opacity-80">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="bg-white text-black p-1.5 rounded-sm border border-white">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tighter">SENTINEL<span className="text-accent-500">.AI</span></h1>
            <p className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest">Intrusion Detection System</p>
          </div>
        </div>

        <div className="flex items-center gap-3 bg-zinc-950 px-3 py-1.5 rounded-sm border border-zinc-900">
          <Server className="w-3.5 h-3.5 text-zinc-600" />
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-white' : 'bg-accent-500 animate-pulse'}`}></span>
            <span className={`text-[10px] font-bold uppercase tracking-widest ${isOnline ? 'text-zinc-400' : 'text-accent-500'}`}>
              {isOnline ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};