import yaml
import torch
import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ETFChatbot:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        base_model_path = os.path.abspath(self.config["base_model"])
        lora_model_path = os.path.abspath(self.config["lora_model"])

        if not os.path.isdir(base_model_path):
            raise ValueError(f"[ERROR] Base model path does not exist: {base_model_path}")

        if not os.path.isdir(lora_model_path):
            raise ValueError(f"[ERROR] LoRA model path does not exist: {lora_model_path}")

        print("[INFO] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

        print("[INFO] Loading base model in 4-bit mode...")
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

        print("[INFO] Applying LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=torch.float16)
        self.model.eval()

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
        prompt = f"[Prompt] {message} [/Prompt] [Answer]"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Set stopping token
        stop_token_id = self.tokenizer.convert_tokens_to_ids("[/Answer]")
        pad_token_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_kwargs,
                eos_token_id=stop_token_id,
                pad_token_id=pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the part between [Answer] and [/Answer]
        response_clean = response.split("[Answer]")[-1].split("[/Answer]")[0].strip()
        return response_clean

