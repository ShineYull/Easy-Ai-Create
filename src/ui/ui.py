import gradio as gr
from src.ui.ui_handler import UIHandler

class UIManage:

    def __init__(self):
        print("创建ui")

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