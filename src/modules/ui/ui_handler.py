import torch
import torchaudio
from src.models.audiocraft.models import MusicGen
from src.models.audiocraft.data.audio import audio_write

class UIHandler:

    def text_handler(self, name):
        return "Hello " + name + "!"

    def audio_handler(self, audio, model):
        # model = MusicGen.get_pretrained('melody')
        print("audio:", audio)
        print("model:", model)


        model = MusicGen.get_pretrained(model)

        print("设置生成参数")
        model.set_generation_params(duration=8)
        # wav = model.generate_unconditional(4)
        descriptions = ['happy rock', 'energetic EDM', 'sad jazz']
        # wav = model.generate(descriptions)

        print("加载音乐")
        # melody, sr = torchaudio.load('./assets/bach.wav')
        # print("melody:", melody)
        # melody_modify = melody[None].expand(3, -1, -1)
        # print("melody_modify:", melody_modify)

        
        sr, audio = audio[0], torch.from_numpy(audio[1]).to('cpu').float().t()
        if audio.dim() == 1:
            audio = audio[None]
        print("sr:", sr)
        print("audio1:", audio)
        wav = model.generate_with_chroma(descriptions, audio.expand(3, -1, -1), sr)

        for idx, one_wav in enumerate(wav):
            print("run audio write.")
            path = audio_write(f'output_music/{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            print("path:", path)

    def auth_handler(self, username, password):
        return username == password

    def interrupt(self):
        global INTERRUPTING
        INTERRUPTING = True