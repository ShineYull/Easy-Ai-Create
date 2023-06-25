import gradio as gr
from ui_handler import UIHandler

class UI:
    
    def __init__(self):
        pass

    def ui(self):
        uihandler = UIHandler()
        with gr.Blocks() as interface:
            input_txt = gr.Textbox(label="输入")
            output_txt = gr.Textbox(label="输出")
            btn_txt = gr.Button("提交")
            btn_txt.click(
                fn=uihandler.text_handler,
                inputs=input_txt,
                outputs=output_txt,
                api_name="text_api"
            )
        interface.launch(auth=uihandler.auth_handler, auth_message="username and password must be the same")

    def audio_ui(self):
        uihandler = UIHandler()
        with gr.Blocks() as interface:
            input_audio = gr.Audio(label="输入音频")
            output_audio = gr.Audio(label="输出音频")
            btn_audio = gr.Button("提交")
            btn_audio.click(
                fn=uihandler.audio_handler,
                inputs=input_audio,
                outputs=output_audio,
                api_name="audio_api"
            )