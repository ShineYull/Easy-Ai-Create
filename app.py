import gradio as gr

def text_handle(name):
    return "Hello " + name + "!"

#账户和密码相同就可以通过
def same_auth(username, password):
    return username == password

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
    interface.launch(auth=same_auth,auth_message="username and password must be the same")

if __name__ == "__main__":
    ui()