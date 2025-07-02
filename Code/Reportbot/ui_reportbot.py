import gradio as gr
from pyngrok import ngrok
from reportbot_engine import ETFreportbot

# Load chatbot model
bot = ETFreportbot()

# Start Ngrok tunnel (optional)
port = bot.config.get("port", 7860)
public_url = ngrok.connect(port)
print(f"Shareable chatbot link: {public_url}")

# Define Gradio interface
def chat_interface(user_input):
    if not user_input.strip():
        return "Please enter the ETF name"
    return bot.ask(user_input)

# Create Gradio app
iface = gr.Interface(
    fn=chat_interface,
    inputs=gr.Textbox(lines=3, placeholder="Ask about ETFs..."),
    outputs="text",
    title="ETF Report",
    description="Ask about ETF will return a report of the ETF",
)

# Launch the app
iface.launch(server_port=port, share=True)
