/**
 * High-performance Web Audio API queue player for 24kHz linear PCM audio chunks.
 * Supports seamless gapless playback and instantaneous .stop() flush on barge-in.
 */
export class AudioQueuePlayer {
  private audioContext: AudioContext | null = null;
  private sampleRate = 24000; // Kokoro-82M sample rate
  private nextStartTime = 0;
  private activeNodes: AudioBufferSourceNode[] = [];
  private onPlaybackEndedCallback: (() => void) | null = null;
  private pendingChunksCount = 0;

  constructor(onPlaybackEnded?: () => void) {
    if (onPlaybackEnded) {
      this.onPlaybackEndedCallback = onPlaybackEnded;
    }
  }

  private initContext(): AudioContext {
    if (!this.audioContext || this.audioContext.state === 'closed') {
      this.audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({
        sampleRate: this.sampleRate,
      });
      this.nextStartTime = 0;
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    return this.audioContext;
  }

  /**
   * Enqueue a 24kHz Int16 linear PCM chunk for seamless playback.
   */
  queueAudioChunk(pcm16ArrayBuffer: ArrayBuffer): void {
    const ctx = this.initContext();
    const int16Array = new Int16Array(pcm16ArrayBuffer);
    const float32Array = new Float32Array(int16Array.length);

    // Convert Int16 -> Float32 [-1.0, 1.0]
    for (let i = 0; i < int16Array.length; i++) {
      float32Array[i] = int16Array[i] / (int16Array[i] < 0 ? 32768 : 32767);
    }

    const audioBuffer = ctx.createBuffer(1, float32Array.length, this.sampleRate);
    audioBuffer.copyToChannel(float32Array, 0);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    const currentTime = ctx.currentTime;
    // Schedule seamlessly right after the previous chunk, or immediately if falling behind
    const startTime = Math.max(currentTime, this.nextStartTime);
    source.start(startTime);
    this.nextStartTime = startTime + audioBuffer.duration;

    this.activeNodes.push(source);
    this.pendingChunksCount++;

    source.onended = () => {
      const idx = this.activeNodes.indexOf(source);
      if (idx !== -1) {
        this.activeNodes.splice(idx, 1);
      }
      this.pendingChunksCount--;
      if (this.pendingChunksCount <= 0) {
        this.pendingChunksCount = 0;
        if (this.onPlaybackEndedCallback) {
          this.onPlaybackEndedCallback();
        }
      }
    };
  }

  /**
   * INSTANTANEOUS FLUSH: Stops all queued and currently playing audio nodes within <10ms.
   * Triggered immediately on barge-in / user interruption.
   */
  flush(): void {
    for (const node of this.activeNodes) {
      try {
        node.stop(0);
        node.disconnect();
      } catch {
        // Node might already have completed
      }
    }
    this.activeNodes = [];
    this.pendingChunksCount = 0;
    if (this.audioContext) {
      this.nextStartTime = this.audioContext.currentTime;
    }
  }

  close(): void {
    this.flush();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}
