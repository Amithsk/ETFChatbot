import gradio as gr
from pyngrok import ngrok
from chatbot_engine import ETFChatbot

# Load chatbot model
bot = ETFChatbot()

# Start Ngrok tunnel
public_url = ngrok.connect(bot.config["port"])
print(f"🔗 Shareable chatbot link: {public_url}")

# Define Gradio interface
def chat_interface(user_input):
    return bot.ask(user_input)

iface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about ETFs..."),
    outputs="text",
    title="ETF Chatbot (TinyLLaMA)",
    description="Ask me about ETF returns, tracking error, AUM, and more.",
)

# Launch
iface.launch(server_port=bot.config["port"], share=False)
