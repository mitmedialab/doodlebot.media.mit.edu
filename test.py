import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import time

def record_audio(output_filename="test.wav", record_seconds=5):
    # Audio parameters
    sample_rate = 16000
    channels = 1
    
    print("Recording will start in 3 seconds...")
    time.sleep(3)
    print("🎤 Recording...")
    
    # Record audio
    recording = sd.rec(
        int(record_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype=np.int16
    )
    
    # Show progress bar
    for i in range(record_seconds):
        progress = (i + 1) / record_seconds
        print(f"\rProgress: [{'=' * int(50 * progress)}{' ' * (50 - int(50 * progress))}] {int(progress * 100)}%", end='')
        time.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    print("\n✅ Finished recording!")
    
    # Save as WAV
    wav.write(output_filename, sample_rate, recording)
    print(f"✅ Audio saved as {output_filename}")
    
    return output_filename

def play_audio(filename):
    print(f"🔊 Playing {filename}...")
    # Read the WAV file
    sample_rate, data = wav.read(filename)
    
    # Play the audio
    sd.play(data, sample_rate)
    # Wait until the audio is finished
    sd.wait()
    print("✅ Playback complete!")

if __name__ == "__main__":
    # Record 5 seconds of audio and save as test.wav
    recorded_file = record_audio("test.wav", record_seconds=5)
    
    # Ask if user wants to play it back
    response = input("\nWould you like to play back the recording? (y/n): ")
    if response.lower() == 'y':
        play_audio(recorded_file)