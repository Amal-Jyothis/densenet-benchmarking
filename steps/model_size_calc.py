import os

def model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    model_size_mb = size_bytes / (1024 ** 2)

    return model_size_mb