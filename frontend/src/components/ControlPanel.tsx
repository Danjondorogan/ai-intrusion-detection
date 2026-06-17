import React, { useState, useEffect } from 'react';
import {
  Play,
  Square,
  Wand2,
  AlertCircle,
  Shield,
  ShieldAlert,
  ShieldQuestion
} from 'lucide-react';
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

const generateNormalFeatures = () => {
  return Array.from(
    { length: FEATURE_COUNT },
    () => (Math.random() * 100).toFixed(4)
  ).join(', ');
};

const generateSuspiciousFeatures = () => {
  return Array.from(
    { length: FEATURE_COUNT },
    () => (Math.random() * 5000000).toFixed(0)
  ).join(', ');
};

const generateAttackFeatures = () => {
  return Array.from(
    { length: FEATURE_COUNT },
    () => (Math.random() * 100000000).toFixed(0)
  ).join(', ');
};

const generateRandomFeatures = () => {
  const mode = Math.floor(Math.random() * 3);

  if (mode === 0) return generateNormalFeatures();
  if (mode === 1) return generateSuspiciousFeatures();

  return generateAttackFeatures();
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
    const values = featuresRaw
      .split(/[\s,]+/)
      .filter(v => v.trim() !== '')
      .map(Number);

    setFeatureCount(values.length);

    setIsValid(
      values.length === FEATURE_COUNT &&
      !values.some(isNaN)
    );
  }, [featuresRaw]);

  const toggleAutoSend = () => {
    if (isValid && sessionId) {
      setIsAutoSending(!isAutoSending);
    }
  };

  return (
    <div className="bg-zinc-950 rounded-sm border border-zinc-900 p-6 flex flex-col h-full">

      <div className="flex items-center justify-between mb-6">
        <h3 className="text-white font-medium text-sm tracking-wide uppercase">
          Simulation Control
        </h3>

        <div
          className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-widest border ${
            isAutoSending
              ? 'bg-red-900/20 border-red-500 text-red-400'
              : 'bg-zinc-900 border-zinc-800 text-zinc-500'
          }`}
        >
          {isAutoSending ? 'AUTO MODE' : 'MANUAL'}
        </div>
      </div>

      {/* SESSION */}

      <div className="mb-5">
        <label className="block text-zinc-500 text-[10px] font-bold uppercase tracking-widest mb-2">
          Session ID
        </label>

        <input
          type="text"
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
          placeholder="session_alpha"
          className="w-full bg-black border border-zinc-800 rounded-sm px-4 py-3 text-white font-mono text-sm focus:border-white outline-none"
        />
      </div>

      {/* FEATURE VECTOR */}

      <div className="mb-5 flex-grow flex flex-col">
        <div className="flex justify-between items-center mb-2">
          <label className="text-zinc-500 text-[10px] font-bold uppercase tracking-widest">
            Feature Vector
          </label>

          <span
            className={`text-[10px] font-mono px-2 py-0.5 rounded-sm ${
              isValid
                ? 'text-green-400 bg-green-950/20'
                : 'text-red-400 bg-red-950/20'
            }`}
          >
            {featureCount} / {FEATURE_COUNT}
          </span>
        </div>

        <textarea
          value={featuresRaw}
          onChange={(e) => setFeaturesRaw(e.target.value)}
          placeholder={`Paste ${FEATURE_COUNT} Features Here`}
          className={`w-full flex-grow bg-black border rounded-sm p-3 text-[10px] text-zinc-300 font-mono resize-none outline-none ${
            isValid
              ? 'border-zinc-800'
              : 'border-red-900'
          }`}
        />

        {!isValid && featureCount > 0 && (
          <p className="text-red-400 text-[10px] mt-2 flex items-center gap-1 font-mono">
            <AlertCircle className="w-3 h-3" />
            INVALID FEATURE COUNT
          </p>
        )}
      </div>

      {/* GENERATORS */}

      <div className="grid grid-cols-2 gap-2 mb-4">

        <button
          onClick={() => setFeaturesRaw(generateNormalFeatures())}
          className="bg-green-950/20 hover:bg-green-900/30 border border-green-900 text-green-400 py-2 rounded-sm text-xs font-bold uppercase flex items-center justify-center gap-2"
        >
          <Shield className="w-3 h-3" />
          Normal
        </button>

        <button
          onClick={() => setFeaturesRaw(generateSuspiciousFeatures())}
          className="bg-yellow-950/20 hover:bg-yellow-900/30 border border-yellow-900 text-yellow-400 py-2 rounded-sm text-xs font-bold uppercase flex items-center justify-center gap-2"
        >
          <ShieldQuestion className="w-3 h-3" />
          Suspicious
        </button>

        <button
          onClick={() => setFeaturesRaw(generateAttackFeatures())}
          className="bg-red-950/20 hover:bg-red-900/30 border border-red-900 text-red-400 py-2 rounded-sm text-xs font-bold uppercase flex items-center justify-center gap-2"
        >
          <ShieldAlert className="w-3 h-3" />
          Attack
        </button>

        <button
          onClick={() => setFeaturesRaw(generateRandomFeatures())}
          className="bg-zinc-900 hover:bg-zinc-800 border border-zinc-700 text-white py-2 rounded-sm text-xs font-bold uppercase flex items-center justify-center gap-2"
        >
          <Wand2 className="w-3 h-3" />
          Random
        </button>

      </div>

      {/* ACTION BUTTONS */}

      <div className="grid grid-cols-2 gap-3">

        <button
          onClick={onSend}
          disabled={
            !isValid ||
            !sessionId ||
            isAutoSending ||
            isLoading
          }
          className="bg-white hover:bg-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed text-black py-3 rounded-sm text-xs font-bold uppercase tracking-wider"
        >
          {isLoading ? 'Sending...' : 'Send Once'}
        </button>

        <button
          onClick={toggleAutoSend}
          disabled={!isValid || !sessionId}
          className={`py-3 rounded-sm text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 ${
            isAutoSending
              ? 'bg-red-500 text-white'
              : 'border border-zinc-700 text-zinc-300 hover:border-white'
          }`}
        >
          {isAutoSending ? (
            <>
              <Square className="w-3 h-3 fill-current" />
              Stop
            </>
          ) : (
            <>
              <Play className="w-3 h-3 fill-current" />
              Auto
            </>
          )}
        </button>

      </div>

    </div>
  );
};