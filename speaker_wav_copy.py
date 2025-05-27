import sounddevice as sd
import soundfile as sf

def record_speaker_wav(filename="reference_voice.wav", duration=5, fs=16000):
    print("🎤 Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    sf.write(filename, recording, fs)  # Write using soundfile
    print(f"✅ Saved as {filename}")
    return filename