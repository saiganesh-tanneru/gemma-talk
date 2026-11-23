import React from 'react';
import { AssistantState } from '../hooks/useVoiceAssistant';
import { Mic, Cpu, Volume2, AlertCircle, Radio, PauseCircle } from 'lucide-react';

interface StatusBadgeProps {
  state: AssistantState;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ state }) => {
  const getBadgeConfig = () => {
    switch (state) {
      case 'LISTENING':
        return {
          label: 'Listening...',
          color: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
          dotColor: 'bg-emerald-400',
          icon: <Mic className="w-4 h-4 animate-pulse text-emerald-400" />,
        };
      case 'TRANSCRIBING':
        return {
          label: 'Transcribing speech',
          color: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
          dotColor: 'bg-blue-400',
          icon: <Radio className="w-4 h-4 animate-spin text-blue-400" />,
        };
      case 'THINKING':
        return {
          label: 'Generating response',
          color: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
          dotColor: 'bg-purple-400',
          icon: <Cpu className="w-4 h-4 animate-pulse text-purple-400" />,
        };
      case 'SPEAKING':
        return {
          label: 'Speaking (Barge-in active)',
          color: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
          dotColor: 'bg-amber-400',
          icon: <Volume2 className="w-4 h-4 animate-bounce text-amber-400" />,
        };
      case 'INTERRUPTED':
        return {
          label: 'Barged in! Cut off speech',
          color: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
          dotColor: 'bg-rose-400',
          icon: <AlertCircle className="w-4 h-4 text-rose-400" />,
        };
      case 'IDLE':
      default:
        return {
          label: 'Ready / Standby',
          color: 'bg-neutral-800/60 text-neutral-400 border-neutral-700/50',
          dotColor: 'bg-neutral-500',
          icon: <PauseCircle className="w-4 h-4 text-neutral-400" />,
        };
    }
  };

  const config = getBadgeConfig();

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: '0.4rem 0.9rem',
        borderRadius: '9999px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        backgroundColor: 'rgba(20, 20, 25, 0.8)',
        backdropFilter: 'blur(8px)',
        fontSize: '0.875rem',
        fontWeight: 500,
      }}
    >
      <span
        style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          backgroundColor:
            state === 'LISTENING'
              ? '#10b981'
              : state === 'SPEAKING'
              ? '#f59e0b'
              : state === 'THINKING' || state === 'TRANSCRIBING'
              ? '#8b5cf6'
              : state === 'INTERRUPTED'
              ? '#ef4444'
              : '#71717a',
          boxShadow: state !== 'IDLE' ? '0 0 10px currentColor' : 'none',
        }}
      />
      {config.icon}
      <span style={{ color: '#f4f4f5' }}>{config.label}</span>
    </div>
  );
};
