import gradio as gr
from chatbot_engine import ETFChatbot

chatbot = ETFChatbot("Code/Chatbot/config.yaml")

def user_input(message, chat_history):
    response = chatbot.ask(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# Indian ETF Chatbot")
    chatbot_ui = gr.Chatbot(label="Chatbot", type="messages")
    txt = gr.Textbox(placeholder="Ask about Indian ETF returns...", container=False)

    txt.submit(user_input, [txt, chatbot_ui], [txt, chatbot_ui])

demo.launch(share=True)
