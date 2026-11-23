import React, { useRef, useEffect } from 'react';
import { useVoiceAssistant } from './hooks/useVoiceAssistant';
import { StatusBadge } from './components/StatusBadge';
import { LatencyHud } from './components/LatencyHud';
import { PhoneCall, PhoneOff, Sparkles, ShieldCheck, Cpu, Mic } from 'lucide-react';

export const App: React.FC = () => {
  const {
    state,
    isConnected,
    messages,
    telemetry,
    micVolume,
    error,
    startCall,
    stopCall,
  } = useVoiceAssistant();

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Dynamic orb scaling based on mic volume when listening
  const volumeScale = isConnected && state === 'LISTENING' ? 1 + (micVolume / 100) * 0.45 : 1;

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="brand-logo">
            <Sparkles className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h1 className="brand-title">Gemma-Talk</h1>
            <p className="brand-subtitle">100% Local Real-Time Conversational Voice Assistant</p>
          </div>
        </div>

        <div className="chips">
          <span className="chip">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" /> Apple M2 Local
          </span>
          <span className="chip">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> AI4Bharat Telugu (te)
          </span>
          <span className="chip">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" /> Ollama gemma3:4b
          </span>
        </div>
      </header>

      {/* Main Interactive Stage */}
      <main className="stage">
        {/* State Badge */}
        <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <StatusBadge state={state} />
          {isConnected && (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.75rem',
                padding: '0.35rem 0.65rem',
                borderRadius: '9999px',
                background: micVolume > 5 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                color: micVolume > 5 ? '#34d399' : '#71717a',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                transition: 'all 0.15s ease',
              }}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Mic: {micVolume}%</span>
            </div>
          )}
        </div>

        {/* Central Glowing Orb & Toggle Button */}
        <div className="orb-wrapper" style={{ transform: `scale(${volumeScale})`, transition: 'transform 0.1s ease-out' }}>
          <div
            className={`orb-glow ${
              isConnected
                ? state === 'SPEAKING'
                  ? 'orb-speaking'
                  : state === 'THINKING' || state === 'TRANSCRIBING'
                  ? 'orb-thinking'
                  : state === 'INTERRUPTED'
                  ? 'orb-interrupted'
                  : 'orb-listening'
                : 'orb-idle'
            }`}
          />
          <button
            onClick={isConnected ? stopCall : startCall}
            className={`call-button ${isConnected ? 'call-active' : 'call-start'}`}
            title={isConnected ? 'End Call' : 'Start Hands-Free Call'}
          >
            {isConnected ? (
              <PhoneOff className="w-8 h-8 text-rose-300" />
            ) : (
              <PhoneCall className="w-8 h-8 text-white" />
            )}
          </button>
        </div>

        {/* Call Status Description */}
        <div className="call-info">
          <h2 className="call-status">
            {isConnected ? 'Hands-Free Call Active' : 'Ready to Connect'}
          </h2>
          <p className="call-hint">
            {isConnected
              ? 'Speak naturally at any time. Barge in anytime while the AI is talking to interrupt.'
              : 'Click to start continuous conversational voice (No Push-To-Talk).'}
          </p>
        </div>

        {/* Telemetry HUD */}
        <div style={{ width: '100%', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
          <LatencyHud telemetry={telemetry} />
        </div>

        {/* Error Alert */}
        {error && (
          <div className="error-banner">
            <span>{error}</span>
          </div>
        )}

        {/* Live Conversation Transcript Feed */}
        <div className="transcript-box">
          <div className="transcript-header">
            <span>Real-Time Transcript</span>
            <span className="transcript-badge">
              {messages.length} turn{messages.length === 1 ? '' : 's'}
            </span>
          </div>

          <div className="transcript-list">
            {messages.length === 0 ? (
              <div className="transcript-empty">
                {isConnected
                  ? 'Listening to your voice... Say "Hello!" or ask a question.'
                  : 'No conversation turns yet. Click the call button above to begin.'}
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`message-bubble ${
                    msg.role === 'user' ? 'message-user' : 'message-assistant'
                  }`}
                >
                  <div className="message-sender">
                    {msg.role === 'user' ? 'You' : 'Gemma Assistant'}
                    {msg.language && (
                      <span style={{ marginLeft: '0.4rem', opacity: 0.7, textTransform: 'uppercase', fontSize: '0.65rem' }}>
                        [{msg.language}]
                      </span>
                    )}
                  </div>
                  <div className="message-text">{msg.text}</div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </div>
      </main>
    </div>
  );
};
