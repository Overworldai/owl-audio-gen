def get_model_cls(model_id):
    if model_id == "audio":
        from .audio import AudioDiffusionModel
        return AudioDiffusionModel
    else:
        raise ValueError(f"Invalid model ID: {model_id}")
