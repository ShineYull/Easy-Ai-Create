import torch
import gradio as gr

from modules.loader.loaders import LoadModels

HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "melody": "facebook/musicgen-melody",
    "large": "facebook/musicgen-large",
    "medium": "facebook/musicgen-medium"
}

HF_MODEL_CP_STATE_MAP = {
    "state": "state_dict.bin",
    "compression": "compression_state_dict.bin"
}

class MusicGen:
    
    def __init__(self):
        pass

    @staticmethod
    def get_pretrained(name: str = 'melody', device = None, repo_id=0, state='state', cache_dir='./'):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        load_model = LoadModels()
        return load_model.download_models(
            repo_id=HF_MODEL_CHECKPOINTS_MAP[name], 
            filename=HF_MODEL_CP_STATE_MAP[state], 
            cache_dir=cache_dir
        )

    @staticmethod
    def do_predictions():
        pass

    @staticmethod
    def predict_full(model, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
        global INTERRUPTING
        INTERRUPTING = False
        if temperature < 0:
            raise gr.Error("Temperature must be >= 0.")
        if topk < 0:
            raise gr.Error("Topk must be non-negative.")
        if topp < 0:
            raise gr.Error("Topp must be non-negative.")

        topk = int(topk)
        
        load_model = LoadModels()
        load_model(model)

        outs = MusicGen.do_predictions(
            [text], [melody], duration, progress=True,
            top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
        return outs[0]