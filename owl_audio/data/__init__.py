def get_loader(data_id, batch_size, **kwargs):
    if data_id == "audio_dir_loader":
        from .audio_dir_loader import get_loader as audio_dir_loader
        return audio_dir_loader(batch_size, **kwargs)
    if data_id == "audio_video_loader":
        from .audio_video_loader import get_loader as audio_video_loader
        return audio_video_loader(batch_size, **kwargs)
    if data_id == "audio_video_txt_loader":
        from .audio_video_txt_loader import get_loader as audio_video_txt_loader
        return audio_video_txt_loader(batch_size, **kwargs)
    raise ValueError(f"Data loader {data_id} not found")
