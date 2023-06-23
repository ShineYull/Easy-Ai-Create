import gradio as gr

def text_handle(name):
    return "Hello " + name + "!"

def ui():
    with gr.Blocks() as interface:
        input_txt = gr.Textbox(label="输入")
        output_txt = gr.Textbox(label="输出")
        btn_txt = gr.Button("提交")
        btn_txt.click(
            fn=text_handle,
            inputs=input_txt,
            outputs=output_txt,
            api_name="text_api"
        )
    interface.launch()

if __name__ == "__main__":
    ui()