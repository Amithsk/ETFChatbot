import yaml
from llama_cpp import Llama
from pathlib import Path


class ETFreportbot:
    def __init__(self, config_filename="config.yaml"):
        base_dir = Path(__file__).parent.resolve()
        config_path = base_dir / config_filename

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Convert model_path to absolute if it's not already
        model_path = Path(self.config["model_path"])
        if not model_path.is_absolute():
            model_path = (base_dir / model_path).resolve()

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=self.config.get("n_ctx", 2048),
            n_threads=4,
            n_gpu_layers=self.config.get("n_gpu_layers", 0)
        )

    def ask(self, user_input: str) -> str:
        try:
            system_prompt = self.config.get("system_prompt", "").strip()
            full_prompt = f"{system_prompt}\n\nETF Query: {user_input}\n\nETF Report:"
        
            response = self.llm(
                full_prompt,
                max_tokens=self.config.get("max_tokens", 512),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.95),
                stop=["</s>"]  # Optional: stop token for cleaner endings
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return f"Error: {str(e)}"
        



