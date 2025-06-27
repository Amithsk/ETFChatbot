import gradio as gr
from chatbot_engine import ETFChatbot

chatbot = ETFChatbot("Chatbot/config.yaml")

def respond(message, history):
    answer = chatbot.ask(message)
    return answer

with gr.Blocks() as demo:
    gr.Markdown("## Indian ETF Chatbot")
    chat = gr.ChatInterface(fn=respond)

if __name__ == "__main__":
    demo.launch(share=True)
