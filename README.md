Create a directory named 'models' and download models from the following links to save in them. 
VOSK folder, extract from zip:
https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

LLAMA3 gguf:
https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf 

LLAMA.CPP gguf:
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

<img width="858" height="122" alt="image" src="https://github.com/user-attachments/assets/5182e7bf-dd6c-4e7e-9805-0a7b89288865" />

Vosk is to be added in /models/vosk/... as shown in image.

- use python version 3.10
- 13.py and 15.py:  - stt: vosk
                    - llm: llama3
                    - tts: xtts

- 10.py:  - stt: vosk
          - llm: llama-cpp
          - tts: xtts

  THE PROJECT REQUIRES MORE THAN 4GB GPU(vram) to run.
  
Latest file is email+task.py which handles and detects user intent, from setting reminders, to sending emails, this agent will handle it all!
