def get_trainer_cls(trainer_id):
    if trainer_id == "audio":
        from .audio import AudioTrainer
        return AudioTrainer
    else:
        raise ValueError(f"Invalid trainer ID: {trainer_id}")
