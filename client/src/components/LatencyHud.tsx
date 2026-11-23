import React from 'react';
import { TelemetryMetrics } from '../hooks/useVoiceAssistant';
import { Zap, Activity } from 'lucide-react';

interface LatencyHudProps {
  telemetry: TelemetryMetrics | null;
}

export const LatencyHud: React.FC<LatencyHudProps> = ({ telemetry }) => {
  if (!telemetry) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.5rem',
          padding: '0.6rem 1rem',
          borderRadius: '12px',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.07)',
          color: '#71717a',
          fontSize: '0.8rem',
        }}
      >
        <Activity className="w-3.5 h-3.5" />
        <span>Telemetry ready (speaks to measure turn latency)</span>
      </div>
    );
  }

  const isSubTarget = telemetry.total_e2e_ms <= 1500;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '0.75rem',
        width: '100%',
        maxWidth: '680px',
        margin: '0 auto',
      }}
    >
      <div className="metric-card">
        <span className="metric-label">STT (Whisper)</span>
        <span className="metric-value">{telemetry.stt_ms} ms</span>
      </div>

      <div className="metric-card">
        <span className="metric-label">LLM TTFT</span>
        <span className="metric-value">{telemetry.ttft_ms} ms</span>
      </div>

      <div className="metric-card">
        <span className="metric-label">TTS 1st Chunk</span>
        <span className="metric-value">{telemetry.tts_first_chunk_ms} ms</span>
      </div>

      <div
        className="metric-card"
        style={{
          borderColor: isSubTarget ? 'rgba(16, 185, 129, 0.4)' : 'rgba(245, 158, 11, 0.4)',
          background: isSubTarget ? 'rgba(16, 185, 129, 0.08)' : 'rgba(245, 158, 11, 0.08)',
        }}
      >
        <span className="metric-label" style={{ color: isSubTarget ? '#34d399' : '#fbbf24' }}>
          <Zap className="w-3 h-3 inline mr-1" /> Total E2E
        </span>
        <span className="metric-value" style={{ color: isSubTarget ? '#10b981' : '#f59e0b' }}>
          {telemetry.total_e2e_ms} ms
        </span>
      </div>
    </div>
  );
};
