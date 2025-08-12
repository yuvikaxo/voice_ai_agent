import os
import wave
import numpy as np
import pyaudio
import torch
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from llama_cpp import Llama
from TTS.api import TTS
import json
import threading
import time
from collections import deque
import re
import torch.serialization



# paths to models
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
LLM_MODEL_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  
VOICE_SAMPLE_PATH = "my_voice.wav"

# audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# performance settings
SILENCE_TIMEOUT = 2.0  
RESPONSE_MAX_TOKENS = 150  
MIN_SPEECH_LENGTH = 0.5  
CONVERSATION_MEMORY = 4  

# convo memory
class ConversationMemory:
    def __init__(self, max_size=CONVERSATION_MEMORY):
        self.messages = deque(maxlen=max_size)
        self.context_keywords = set()
    
    def add_exchange(self, user_input, ai_response):
        """Add a user-AI exchange to memory"""
        self.messages.append({
            'user': user_input,
            'ai': ai_response,
            'timestamp': time.time()
        })
        words = re.findall(r'\b\w+\b', user_input.lower())
        self.context_keywords.update(words)
    
    def get_context(self):
        """Get recent conversation context"""
        if not self.messages:
            return ""
        
        context = ""
        for msg in list(self.messages)[-2:]:  
            context += f"User: {msg['user']}\nAI: {msg['ai']}\n"
        return context
    
    def is_followup_question(self, user_input):
        """Check if this seems like a follow-up question"""
        followup_patterns = [
            r'\b(what about|how about|and|also|tell me more|explain|why|how)\b',
            r'\b(that|this|it|they|them)\b',
            r'\b(more|again|further|continue)\b'
        ]
        return any(re.search(pattern, user_input.lower()) for pattern in followup_patterns)

print("Initializing models...")

conversation_memory = ConversationMemory()

# 1. VOSK 
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"VOSK model not found at {VOSK_MODEL_PATH}")
    exit()
vosk_model = Model(VOSK_MODEL_PATH)

# 2. Llama 3 
try:
    # Check available GPU memory
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory Available: {gpu_memory:.1f} GB")
        
        # Conservative GPU layers to stay under 3.7GB
        if gpu_memory >= 4.0:
            n_gpu_layers = 15  # Partial GPU acceleration
        else:
            n_gpu_layers = 0
            print("Low GPU memory, using CPU only")
    else:
        n_gpu_layers = 0
        print("No GPU available, using CPU")
    
    llm = Llama(
        model_path=LLM_MODEL_PATH, 
        n_ctx=4096,  # Llama 3 can handle larger context
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        n_threads=6,  # Optimized for CPU performance
        n_batch=512,  # Balanced batch size
        use_mmap=True,
        use_mlock=False,
        rope_freq_base=500000,  # Llama 3 specific
        rope_freq_scale=1.0     # Llama 3 specific
    )
    print("Llama 3 loaded successfully")
    
    # Test the model with Llama 3 format
    test_response = llm("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", max_tokens=10, temperature=0.7)
    print(f"Llama 3 test: {test_response['choices'][0]['text'].strip()}")
    
except Exception as e:
    print(f"Error initializing Llama 3 model: {e}")
    print("Make sure you have the Llama 3 8B Instruct Q4_K_M model file!")
    exit()

# 3. Coqui XTTS
# Agree to Coqui terms
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from torch.serialization import add_safe_globals

# Register trusted globals for safe PyTorch deserialization
add_safe_globals([XttsConfig, XttsAudioConfig])


# Patch torch.load to force weights_only=False
original_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = unsafe_load

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load TTS model
try:
    from TTS.utils.manage import ModelManager
    from TTS.api import TTS
    import torch

    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manager = ModelManager()
    model_paths = manager.download_model(model_name)

    if not model_paths or model_paths[0] is None or model_paths[1] is None:
        raise ValueError("Model path or config path is None.")

    model_path = model_paths[0]
    config_path = model_paths[1]

    print(f"Model path: {model_path}")
    print(f"Config path: {config_path}")

    global tts
    tts = TTS(model_path=model_path, config_path=config_path).to(device)
    print("TTS loaded successfully")

except Exception as e:
    print("Error loading TTS:", e)

# Check for voice sample file
if not os.path.exists(VOICE_SAMPLE_PATH):
    print(f"Voice sample not found at {VOICE_SAMPLE_PATH}")
    exit()

print("All models ready!")

def listen_for_speech():
    """
    Enhanced speech detection with better audio processing.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    rec = KaldiRecognizer(vosk_model, RATE)
    rec.SetWords(True)

    print("Listening... (speak now)")
    
    recognized_text = ""
    silence_start = None
    speech_detected = False
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    recognized_text = text
                    speech_detected = True
                    silence_start = None
                    print(f"🔊 Recognized: {text}")
                    
                    # Continue listening for more speech
                    time.sleep(0.5)
                    
            else:
                # Check partial results for speech activity
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                
                if partial_text:
                    speech_detected = True
                    silence_start = None
                    print(f"📝 Partial: {partial_text}", end='\r')
                elif speech_detected:
                    # Start silence timer after speech was detected
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_TIMEOUT:
                        # Process any remaining audio
                        final_result = json.loads(rec.FinalResult())
                        final_text = final_result.get("text", "").strip()
                        if final_text and not recognized_text:
                            recognized_text = final_text
                        break
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    if recognized_text:
        print(f"\nYou said: {recognized_text}")
    else:
        print("\nNo speech detected")
    
    return recognized_text

def generate_llm_response(prompt):
    """
    Llama 3 response generation with proper chat formatting.
    """
    # Get context from conversation memory
    context_messages = []
    for msg in list(conversation_memory.messages)[-2:]:  # Last 2 exchanges
        context_messages.append(f"User: {msg['user']}")
        context_messages.append(f"Assistant: {msg['ai']}")
    
    # Build Llama 3 chat format
    chat_prompt = "<|start_header_id|>system<|end_header_id|>\n\n"
    chat_prompt += "You are a helpful voice assistant. Give natural, conversational responses in 1-2 sentences. Be friendly and concise.<|eot_id|>\n"
    
    # Add context if available
    if context_messages:
        for msg in context_messages:
            if msg.startswith("User:"):
                chat_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg[5:].strip()}<|eot_id|>\n"
            else:
                chat_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg[10:].strip()}<|eot_id|>\n"
    
    # Add current user input
    chat_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n"
    chat_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    print("Thinking...")
    print(f"Prompt length: {len(chat_prompt)} characters")
    
    start_time = time.time()
    
    try:
        response = llm(
            chat_prompt, 
            max_tokens=RESPONSE_MAX_TOKENS,
            stop=["<|eot_id|>", "<|end_of_text|>", "\n\n", "<|start_header_id|>"],
            temperature=0.7,  # Good balance for Llama 3
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            echo=False
        )
        
        generation_time = time.time() - start_time
        print(f"Generated in {generation_time:.1f}s")
        
        full_response = response['choices'][0]['text'].strip()
        print(f"Raw response: '{full_response}'")
        
        # Clean up response
        full_response = re.sub(r'^(Assistant:|AI:|Bot:)\s*', '', full_response)
        full_response = full_response.split('\n')[0]  # Take first line only
        
        # Ensure proper sentence ending
        if full_response and not full_response.endswith(('.', '!', '?')):
            # Find last complete sentence
            sentences = re.split(r'(?<=[.!?])\s+', full_response)
            if len(sentences) > 1:
                full_response = ' '.join(sentences[:-1])
            else:
                full_response += '.'
        
        # Fallback responses
        if not full_response or len(full_response) < 3:
            fallbacks = [
                "I'm not sure about that.",
                "Could you clarify what you mean?",
                "I didn't quite understand that.",
                "Can you rephrase that for me?"
            ]
            full_response = fallbacks[len(conversation_memory.messages) % len(fallbacks)]
        
        print(f"AI: {full_response}")
        return full_response
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I had trouble generating a response."

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=True)
VOICE_SAMPLE_PATH = "my_voice.wav"

def speak_response(text):
    """
    Speech synthesis with error handling.
    """
    if not text.strip():
        return
        
    try:
        print("Speaking...")
        start_time = time.time()
        
        # TTS generation
        wav_out = tts.tts(
            text=text, 
            speaker_wav=VOICE_SAMPLE_PATH, 
            language="en", 
            split_sentences=False,
            speed=1.0
        )
        
        tts_time = time.time() - start_time
        print(f"TTS generated in {tts_time:.1f}s")
        
        # Audio normalization
        wav_array = np.array(wav_out)
        if np.max(np.abs(wav_array)) > 0:
            wav_array = wav_array / np.max(np.abs(wav_array)) * 0.8
        
        # Play audio
        sd.play(wav_array, samplerate=24000)
        sd.wait()
        
    except Exception as e:
        print(f"TTS error: {e}")
        # Fallback: just print the text
        print(f"Fallback - Response: {text}")

def debug_model():
    """Debug function to test Llama 3 directly"""
    print("\nDebug Mode - Testing Llama 3 directly...")
    
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke."
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting: {prompt}")
        try:
            # Use proper Llama 3 format for testing
            chat_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            response = llm(
                chat_prompt,
                max_tokens=50,
                temperature=0.7,
                stop=["<|eot_id|>", "<|end_of_text|>"]
            )
            result = response['choices'][0]['text'].strip()
            print(f"Response: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nDebug complete")

def main_loop():
    """
    Enhanced main loop with better error handling and debugging.
    """
    print("Voice Agent ready! (Now with Llama 3!)")
    print("Tips:")
    print("   - Speak clearly and wait for the processing")
    print("   - Say 'debug' to test Llama 3")
    print("   - Say 'exit', 'quit', or 'goodbye' to end")
    print("   - Llama 3 gives much better responses!")
    print("-" * 50)
    
    while True:
        try:
            user_input = listen_for_speech()
            
            if not user_input:
                print("Didn't catch that. Please try again.")
                continue
            
            # Check for debug command
            if "debug" in user_input.lower():
                debug_model()
                continue
            
            # Check for exit commands
            exit_words = ["exit", "quit", "goodbye", "stop", "bye"]
            if any(word in user_input.lower() for word in exit_words):
                print("Goodbye!")
                speak_response("Goodbye! It was nice talking with you.")
                break
            
            # Generate response
            llm_response = generate_llm_response(user_input)
            
            if llm_response:
                # Add to conversation memory
                conversation_memory.add_exchange(user_input, llm_response)
                
                # Speak the response
                speak_response(llm_response)
                
                print("-" * 30)
                print("Ready for your next question...")
            else:
                print("Sorry, I couldn't generate a response. Please try again.")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            print("Continuing...")
            continue

if __name__ == "__main__":
    try:
        # Quick system check
        print(f"System Info:")
        print(f"   - Model file size: {os.path.getsize(LLM_MODEL_PATH) / (1024*1024):.1f} MB")
        print(f"   - PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"   - Voice sample exists: {os.path.exists(VOICE_SAMPLE_PATH)}")
        print("")
        
        main_loop()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        print("Cleaning up...")
        try:
            sd.stop()
        except:
            pass
        print("Done!")