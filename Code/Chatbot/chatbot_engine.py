import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ETFChatbot:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_path = self.config["model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.generation_kwargs = {
            "max_new_tokens": self.config.get("n_ctx", 512),
            "temperature": self.config.get("temperature", 0.7),
            "top_k": self.config.get("top_k", 10),
            "top_p": self.config.get("top_p", 0.9),
            "repetition_penalty": self.config.get("repeat_penalty", 1.2),
        }

        self.system_prompt = self.config.get("system_prompt", "You are a helpful assistant.")

    def ask(self, user_input):
        prompt = f"<s>[Prompt] {user_input} [/Prompt] [Answer]"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(input_ids, **self.generation_kwargs)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "[/Answer]" in decoded:
            return decoded.split("[/Answer]")[-1].strip()
        return decoded.strip()