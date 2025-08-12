import numpy as np
import sounddevice as sd

fs = 44100  # Sample rate
duration = 2  # seconds
frequency = 440.0  # Hz (A4)

# Generate a 2-second sine wave tone
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
wave = 0.5 * np.sin(2 * np.pi * frequency * t)

sd.play(wave, fs)
sd.wait()
