import gradio as gr
from .ui_handler import UIHandler

class UIManage:

    def __init__(self):
        pass

    def ui_full(self):
        uihandler = UIHandler()
        
        with gr.Blocks() as interface:
            with gr.Tab("audio"):
                with gr.Row():
                    model = gr.Radio(["melody", "small", "large", "medium"], value="melody", label="选择模型", interactive=True)
                with gr.Row():
                    audio_time_slider = gr.Slider(label="生成的音频时长", minimum=1, maximum=120, value=10)
                with gr.Row():
                    audio_text = gr.Text(label="对生成的音乐的文本描述")
                with gr.Column(scale=1, min_width=600):
                    with gr.Row():
                        input_audio = gr.Audio(label="输入音频", source="upload", type="numpy", interactive=True)
                    with gr.Row():
                        btn_audio = gr.Button("提交")
                        btn_audio_interrupt = gr.Button("Interrupt")
                        btn_audio_interrupt.click(
                            fn=uihandler.interrupt,
                            queue=False
                        )
                with gr.Row():
                    output_audio = gr.Audio(label="输出音频")
                btn_audio.click(
                    fn=uihandler.audio_handler,
                    inputs=[input_audio, model, audio_time_slider, audio_text],
                    outputs=output_audio,
                    api_name="audio_api"
                )
                gr.Examples(
                    fn=uihandler.audio_handler,
                    examples=[
                        [
                            "An 80s driving pop song with heavy drums and synth pads in the background",
                            "./dataset/audiocraft/bach.wav",
                            "melody"
                        ]
                    ],
                    inputs=[audio_text, input_audio, model],
                    outputs=[output_audio]
                )

            with gr.Tab("text"):
                input_txt = gr.Textbox(label="输入")
                btn_txt = gr.Button("提交")
                output_txt = gr.Textbox(label="输出")
                btn_txt.click(
                    fn=uihandler.text_handler,
                    inputs=input_txt,
                    outputs=output_txt,
                    api_name="text_api"
                )

            with gr.Tab("stable_diffusion"):
                model = gr.Dropdown(["v2-1_768-ema-pruned.ckpt", "xxxxxxx.ckpt", "yyyyyyy.ckpt"], label="选择模型")
                config = gr.Dropdown(["v2-inference-v-mac.yaml", "v2-inference-v.yaml", "v2-inference.yaml"], label="选择配置文件")
                width = gr.Slider(label="生成的图片宽度", minimum=1, maximum=1920, value=512)
                height = gr.Slider(label="生成的图片高度", minimum=1, maximum=1080, value=512)
                prompt_text = gr.Text(label="输入文本提示")
                generate_btn = gr.Button("生成")
                show_img = gr.Image(label="生成的图片")

                generate_btn.click(
                    fn=uihandler.stable_diffusion_handler,
                    inputs=[prompt_text, height, width, config, model],
                    outputs=show_img,
                    api_name="api_stable_diffusion"
                )
                
        interface.launch(auth=uihandler.auth_handler, auth_message="username and password must be the same")