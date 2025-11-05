def get_trainer_cls(trainer_id):
    if trainer_id == "audio_rft":
        from .audio_rft import AudioRFTTrainer
        return AudioRFTTrainer
    else:
        raise ValueError(f"Invalid trainer ID: {trainer_id}")