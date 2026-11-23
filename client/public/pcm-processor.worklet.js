/**
 * AudioWorkletProcessor that downsamples microphone input to 16kHz Int16 PCM
 * and emits fixed 512-sample chunks (32ms per chunk) for Silero VAD.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.chunkSize = 512;
    this.inputBuffer = [];
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];
    // In AudioWorkletGlobalScope, sampleRate is standard
    const currentSampleRate = typeof sampleRate !== 'undefined' ? sampleRate : 48000;

    let resampled;
    if (currentSampleRate === this.targetSampleRate) {
      resampled = channelData;
    } else {
      const ratio = currentSampleRate / this.targetSampleRate;
      const newLength = Math.floor(channelData.length / ratio);
      resampled = new Float32Array(newLength);
      for (let i = 0; i < newLength; i++) {
        const originalIndex = Math.min(Math.floor(i * ratio), channelData.length - 1);
        resampled[i] = channelData[originalIndex];
      }
    }

    for (let i = 0; i < resampled.length; i++) {
      this.inputBuffer.push(resampled[i]);
    }

    while (this.inputBuffer.length >= this.chunkSize) {
      const chunk = this.inputBuffer.splice(0, this.chunkSize);
      const int16Buffer = new Int16Array(this.chunkSize);

      for (let i = 0; i < this.chunkSize; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      this.port.postMessage(int16Buffer.buffer, [int16Buffer.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
