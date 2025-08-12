from TTS.utils.manage import ModelManager

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
manager = ModelManager()
model_path, config_path, model_item = manager.download_model(model_name)
print("Model Path:", model_path)
print("Config Path:", config_path)
