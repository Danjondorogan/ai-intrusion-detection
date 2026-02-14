import React, { useState, useEffect } from 'react';
import { Play, Square, Wand2, AlertCircle } from 'lucide-react';
import { FEATURE_COUNT } from '../types';

interface Props {
  sessionId: string;
  setSessionId: (val: string) => void;
  featuresRaw: string;
  setFeaturesRaw: (val: string) => void;
  onSend: () => void;
  isAutoSending: boolean;
  setIsAutoSending: (val: boolean) => void;
  isLoading: boolean;
}

// Helper to generate 84 random feature values
const generateDummyFeatures = () => {
  return Array.from({ length: FEATURE_COUNT }, () => Math.random().toFixed(6)).join(', ');
};

export const ControlPanel: React.FC<Props> = ({
  sessionId,
  setSessionId,
  featuresRaw,
  setFeaturesRaw,
  onSend,
  isAutoSending,
  setIsAutoSending,
  isLoading
}) => {
  const [featureCount, setFeatureCount] = useState(0);
  const [isValid, setIsValid] = useState(false);

  useEffect(() => {
    // Parse features to count them
    const values = featuresRaw
      .split(/[\s,]+/)
      .filter(v => v.trim() !== '')
      .map(Number);
    setFeatureCount(values.length);
    setIsValid(values.length === FEATURE_COUNT && !values.some(isNaN));
  }, [featuresRaw]);

  const handleAutoFill = () => {
    setFeaturesRaw(generateDummyFeatures());
  };

  const toggleAutoSend = () => {
    if (isValid && sessionId) {
      setIsAutoSending(!isAutoSending);
    }
  };

  return (
    <div className="bg-zinc-950 rounded-sm border border-zinc-900 p-6 flex flex-col h-full">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-white font-medium text-sm tracking-wide uppercase">Simulation</h3>
        <div className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest border ${isAutoSending ? 'bg-accent-900/20 border-accent-500 text-accent-500' : 'bg-zinc-900 border-zinc-800 text-zinc-500'}`}>
            {isAutoSending ? 'AUTO-PILOT' : 'MANUAL'}
        </div>
      </div>

      {/* Session ID */}
      <div className="mb-5">
        <label className="block text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-2">Session ID</label>
        <input 
          type="text" 
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
          className="w-full bg-black border border-zinc-800 rounded-sm px-4 py-3 text-white font-mono text-sm focus:border-white outline-none transition-all placeholder-zinc-700"
          placeholder="SESSION_ID"
        />
      </div>

      {/* Feature Input */}
      <div className="mb-5 flex-grow flex flex-col">
        <div className="flex justify-between items-center mb-2">
            <label className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest">Feature Vector</label>
            <span className={`text-[10px] font-mono px-2 py-0.5 rounded-sm ${isValid ? 'text-zinc-400 bg-zinc-900' : 'text-accent-500 bg-accent-900/20'}`}>
                {featureCount} / {FEATURE_COUNT}
            </span>
        </div>
        <textarea
          value={featuresRaw}
          onChange={(e) => setFeaturesRaw(e.target.value)}
          className={`w-full flex-grow bg-black border rounded-sm p-3 text-[10px] text-zinc-300 font-mono resize-none outline-none focus:border-white transition-all ${
            isValid ? 'border-zinc-800' : 'border-accent-900'
          }`}
          placeholder={`INPUT DATA [${FEATURE_COUNT}]`}
        />
        {!isValid && featureCount > 0 && (
            <p className="text-accent-500 text-[10px] mt-2 flex items-center gap-1 font-mono">
                <AlertCircle className="w-3 h-3" /> 
                INVALID FORMAT
            </p>
        )}
      </div>

      {/* Actions */}
      <div className="grid grid-cols-2 gap-3 mt-auto">
        <button 
          onClick={handleAutoFill}
          className="col-span-2 bg-transparent hover:bg-zinc-900 text-zinc-400 py-3 rounded-sm text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-colors border border-zinc-800 border-dashed"
        >
          <Wand2 className="w-3 h-3" /> Generate Dummy
        </button>
        
        <button
          onClick={onSend}
          disabled={!isValid || !sessionId || isAutoSending || isLoading}
          className="bg-white hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed text-black py-3 rounded-sm text-xs font-bold uppercase tracking-wider transition-colors"
        >
          {isLoading ? '...' : 'Send Single'}
        </button>

        <button
          onClick={toggleAutoSend}
          disabled={!isValid || !sessionId}
          className={`py-3 rounded-sm text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-colors border ${
            isAutoSending 
            ? 'bg-accent-500 hover:bg-accent-600 text-white border-transparent' 
            : 'bg-transparent text-zinc-400 border-zinc-700 hover:border-white hover:text-white'
          }`}
        >
          {isAutoSending ? (
            <><Square className="w-3 h-3 fill-current" /> Stop</>
          ) : (
            <><Play className="w-3 h-3 fill-current" /> Auto</>
          )}
        </button>
      </div>
    </div>
  );
};