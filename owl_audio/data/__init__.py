def get_loader(data, batch_size, **data_kwargs):
    if data == "audio_dir_loader":
        from .audio_dir_loader import get_loader
        return get_loader(batch_size, **data_kwargs)
    else:
        raise ValueError(f"Unknown data id: {data}")
    