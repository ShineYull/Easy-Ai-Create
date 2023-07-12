import torch
import torchaudio
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

    def auth_handler(self, username, password):
        return username == password

    def interrupt(self):
        global INTERRUPTING
        INTERRUPTING = True