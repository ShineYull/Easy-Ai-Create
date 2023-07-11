import torch

from huggingface_hub import hf_hub_download

class LoadModels():

    def __init__(self):
        pass

    def download_models(self, repo_id, filename, cache_dir, device='cpu'):
        file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return torch.load(file, map_location=device)