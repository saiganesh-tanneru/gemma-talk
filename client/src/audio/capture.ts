/**
 * Captures microphone audio with hardware AEC/NS, computes real-time input volume,
 * and delivers 16kHz Int16 linear PCM chunks (512 samples) over WebSocket.
 */
export class AudioCaptureManager {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private scriptNode: ScriptProcessorNode | null = null;
  private onVolumeCallback: ((volume: number) => void) | null = null;

  async start(
    onAudioChunk: (chunk: ArrayBuffer) => void,
    onVolume?: (volume: number) => void
  ): Promise<void> {
    if (onVolume) {
      this.onVolumeCallback = onVolume;
    }

    // 1. Request microphone with browser acoustic hygiene
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    });

    // 2. Initialize AudioContext
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.audioContext = new AudioContextClass();

    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

    // 3. Audio processor buffer (2048 samples per frame)
    const bufferSize = 2048;
    this.scriptNode = this.audioContext.createScriptProcessor(bufferSize, 1, 1);

    const targetSampleRate = 16000;
    const chunkSize = 512;
    let resampleBuffer: number[] = [];

    this.scriptNode.onaudioprocess = (event: AudioProcessingEvent) => {
      const inputBuffer = event.inputBuffer;
      const channelData = inputBuffer.getChannelData(0);
      const inputSampleRate = inputBuffer.sampleRate;
      const ratio = inputSampleRate / targetSampleRate;

      // Mute local output to speakers so user does not hear their own microphone
      const outputData = event.outputBuffer.getChannelData(0);
      outputData.fill(0);

      // Compute RMS volume (0 - 100) for real-time visual feedback
      let sum = 0;
      for (let i = 0; i < channelData.length; i++) {
        sum += channelData[i] * channelData[i];
      }
      const rms = Math.sqrt(sum / channelData.length);
      const volume = Math.min(100, Math.round(rms * 400));
      if (this.onVolumeCallback) {
        this.onVolumeCallback(volume);
      }

      // Resample to 16kHz
      const newLength = Math.floor(channelData.length / ratio);
      for (let i = 0; i < newLength; i++) {
        const originalIndex = Math.min(Math.floor(i * ratio), channelData.length - 1);
        resampleBuffer.push(channelData[originalIndex]);
      }

      // Emit 512-sample Int16 PCM chunks
      while (resampleBuffer.length >= chunkSize) {
        const chunk = resampleBuffer.splice(0, chunkSize);
        const int16Buffer = new Int16Array(chunkSize);
        for (let i = 0; i < chunkSize; i++) {
          const s = Math.max(-1, Math.min(1, chunk[i]));
          int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        onAudioChunk(int16Buffer.buffer);
      }
    };

    this.sourceNode.connect(this.scriptNode);
    this.scriptNode.connect(this.audioContext.destination);
  }

  stop(): void {
    if (this.scriptNode) {
      this.scriptNode.onaudioprocess = null;
      this.scriptNode.disconnect();
      this.scriptNode = null;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.onVolumeCallback = null;
  }
}
