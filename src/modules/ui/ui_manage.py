import gradio as gr
import src.modules.handlers.chatglm_handler as chatglm_handler

from src.modules.handlers.chatglm_handler import predict
from src.modules.handlers.chatglm_handler import reset_user_input
from src.modules.handlers.chatglm_handler import reset_state

from .ui_handler import UIHandler
from transformers import AutoModel, AutoTokenizer

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

            with gr.Tab("Chatbot"):
                # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
                # model: AutoModel = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).float()
                # # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
                # # from utils import load_model_on_gpus
                # # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
                # model = model.eval()
                # gr.Chatbot.postprocess = uihandler.chatbot_postprocess

                chatbot = gr.Chatbot()
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Column(scale=12):
                            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                                container=False)
                        with gr.Column(min_width=32, scale=1):
                            submitBtn = gr.Button("Submit", variant="primary")
                    with gr.Column(scale=1):
                        emptyBtn = gr.Button("Clear History")
                        max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
                        top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                        temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

                history = gr.State([])
                past_key_values = gr.State(None)

                submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                                [chatbot, history, past_key_values], show_progress=True)
                submitBtn.click(reset_user_input, [], [user_input])

                emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

            with gr.Tab("stable_diffusion"):
                model = gr.Dropdown(["v2-1_768-ema-pruned.ckpt", "xxxxxxx.ckpt", "yyyyyyy.ckpt"], label="选择模型")
                config = gr.Dropdown(["v2-inference-v-mac.yaml", "v2-inference-v.yaml", "v2-inference.yaml"], label="选择配置文件")
                width = gr.Slider(label="生成的图片宽度", minimum=1, maximum=1920, value=512)
                height = gr.Slider(label="生成的图片高度", minimum=1, maximum=1080, value=512)
                prompt_text = gr.Text(label="输入文本提示")

                steps = gr.Slider(label="steps", minimum=1, maximum=50, value=30, step=1)
                samples = gr.Slider(label="samples", minimum=1, maximum=3, value=1, step=1)
                iter = gr.Slider(label="iter", minimum=1, maximum=3, value=1, step=1)

                generate_btn = gr.Button("生成")
                show_img = gr.Image(label="生成的图片")

                generate_btn.click(
                    fn=uihandler.stable_diffusion_handler,
                    inputs=[prompt_text, height, width, config, model, steps, samples, iter],
                    outputs=show_img,
                    api_name="api_stable_diffusion"
                )
                gr.Examples(
                    fn=uihandler.stable_diffusion_handler,
                    examples=[
                        [
                            "a professional photograph of an astronaut riding a horse",
                            512,
                            512,
                            "v2-inference-v-mac.yaml",
                            "v2-1_768-ema-pruned.ckpt",
                            30,
                            1,
                            1
                        ]
                    ],
                    inputs=[prompt_text, height, width, config, model, steps, samples, iter],
                    outputs=show_img,
                )

        interface.queue().launch(share=False, inbrowser=True, auth=uihandler.auth_handler, auth_message="username and password must be the same")