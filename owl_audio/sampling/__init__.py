def get_sampler_fn(sampler_id):
    if sampler_id == "audio":
        from .audio import audio_sample
        return audio_sample
    raise ValueError(f"Sampler {sampler_id} not found")
