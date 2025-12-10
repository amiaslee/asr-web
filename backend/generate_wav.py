import wave
import math
import struct

def generate_wav(filename="test_audio.wav", duration=3, frequency=440.0, framerate=16000):
    audio = []
    num_samples = int(duration * framerate)
    
    for i in range(num_samples):
        # Generate a sine wave
        value = int(32767.0 * math.cos(2.0 * math.pi * frequency * i / framerate))
        audio.append(value)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wav_file.setframerate(framerate)
        for sample in audio:
            wav_file.writeframes(struct.pack('h', sample))
    
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_wav()
