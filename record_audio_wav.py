import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(duration, filename, sample_rate=44100):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    write(filename, sample_rate, audio)  # Save as WAV file
    print(f"Recording saved to {filename}")

# Record a 5-second audio and save it as 'output.wav'
record_audio(5, 'output.wav')
