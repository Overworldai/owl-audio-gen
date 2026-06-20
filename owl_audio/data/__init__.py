def get_loader(data_id, batch_size, **kwargs):
    if data_id == "audio_dir_loader":
        from .audio_dir_loader import get_loader as audio_dir_loader
        return audio_dir_loader(batch_size, **kwargs)
    if data_id == "audio_video_loader":
        from .audio_video_loader import get_loader as audio_video_loader
        return audio_video_loader(batch_size, **kwargs)
    if data_id == "video_audio_loader":
        from .video_audio_loader import get_loader as video_audio_loader
        return video_audio_loader(batch_size, **kwargs)
    if data_id == "finevideo":
        from .finevideo_loader import get_loader as finevideo_loader
        return finevideo_loader(batch_size, **kwargs)
    raise ValueError(f"Data loader {data_id} not found")
