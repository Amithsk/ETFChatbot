import yaml
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

class ETFChatbot:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        model_path = os.path.abspath(self.config["model_path"])

        if not os.path.isdir(model_path):
            raise ValueError(f"[ERROR] Model path does not exist: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

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

    def ask(self, message):
        prompt = f"<s>[Prompt] {message} [/Prompt] [Answer]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Trim until answer 
        return response.split("[Answer]")[-1].strip().replace("</s>", "")
