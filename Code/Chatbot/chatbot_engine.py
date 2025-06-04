import yaml
from llama_cpp import Llama

class ETFChatbot:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.llm = Llama(
            model_path=self.config["model_path"],
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=20  # adjust based on your GPU
        )

    def ask(self, prompt: str) -> str:
        try:
            response = self.llm(
                prompt,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error: {str(e)}"
