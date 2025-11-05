def get_model_cls(model_id):
    if model_id == "audio_rft":
        from .audio_rft import AudioRFT
        return AudioRFT
    else:
        raise ValueError(f"Invalid model ID: {model_id}")