from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
tts.tts_to_file(
    text="Hello, this is a test of the voice engine.",
    speaker_wav=r"C:\Users\LENOVO\Desktop\ai-voice-agent-ntcc\my_voice.wav",  # Replace with an actual WAV path
    language="en",
    file_path="output.wav"
)
