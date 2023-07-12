import torch
import torchaudio
import src.models.stable_diffusion.scripts.txt2img as txt2img

from src.models.audiocraft.models import MusicGen
from src.models.audiocraft.data.audio import audio_write

class UIHandler:

    def text_handler(self, name):
        return "Hello " + name + "!"

    def audio_handler(self, audio, model, duration, descriptions):
        model = MusicGen.get_pretrained(model)
        model.set_generation_params(duration=duration)
        sr, audio = audio[0], torch.from_numpy(audio[1]).to('cpu').float().t()
        if audio.dim() == 1:
            audio = audio[None]
        wav = model.generate_with_chroma([descriptions], audio.expand(1, -1, -1), sr)
        path = audio_write(f'output_music/test', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        return path

    def stable_diffusion_handler(self, prompt, img_height, img_width, config, model):
        # opt = txt2img.parse_args()
        result = txt2img.main({
            "C":4, 
            "H":img_height, 
            "W":img_width, 
            "bf16":False, 
            "ckpt":'src/models/stable_diffusion/checkpoints/' + model, 
            "config":'src/models/stable_diffusion/configs/stable-diffusion/' + config, 
            "ddim_eta":0.0, 
            "device":'cpu', 
            "dpm":False, 
            "f":8, 
            "fixed_code":False, 
            "from_file":None, 
            "ipex":False, 
            "n_iter":3, 
            "n_rows":0, 
            "n_samples":3, 
            "outdir":'outputs/txt2img-samples', 
            "plms":False, 
            "precision":'full', 
            "prompt":prompt, 
            "repeat":1, 
            "scale":9.0, 
            "seed":42, 
            "steps":50, 
            "torchscript":False
        })
        
        return result

    def auth_handler(self, username, password):
        return username == password

    def interrupt(self):
        global INTERRUPTING
        INTERRUPTING = True