def get_trainer_cls(trainer_id):
    if trainer_id == "audio":
        from .audio import AudioTrainer
        return AudioTrainer
    if trainer_id == "audio_video":
        from .audio_video import AudioVideoTrainer
        return AudioVideoTrainer
    else:
        raise ValueError(f"Invalid trainer ID: {trainer_id}")
