# ===== train_lora_adapters.py =====
"""
This module defines the high-level pipeline orchestration:
 - Splitting data into chunks
 - Invoking the fine-tuning function on each chunk
 - Consolidating the final adapter
It relies on `train_model_utils.py` for the actual LoRA chunk fine-tuning.
"""
import os
import datetime
import shutil
from zoneinfo import ZoneInfo

# Local imports
from utils.train_model_utils import fine_tune_chunk
from Prompt.etfexpenseratioPromptReturn import generate_expense_ratio_pairs
from Prompt.etfreturnPromptReturn import generate_prompt_response_return_pairs
from utils.train_modelDB_utils import fetch_etf_expense_ratios, fetch_etf_returns

# Configuration constants
RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Temporary adapters live under Models/Training/ModelTraining/<timestamp>
MODEL_ROOT = os.path.join("Models", "Training", "ModelTraining", RUN_TS)
# Final consolidated adapters go under Models/Training/Mistral-LoRA-<timestamp>
FINAL_ROOT = os.path.join("Models", "Training", f"Mistral-LoRA-{RUN_TS}")
LOG_ROOT = "logs"
CHUNK_SIZE = 5000  # examples per chunk
STOP_HOUR = 21  # 9 PM IST
TIMEZONE = ZoneInfo("Asia/Kolkata")

# Ensure base directories
os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(FINAL_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)


def list_checkpoints(output_dir):
    """Return sorted list of checkpoint dirs inside output_dir."""
    if not os.path.isdir(output_dir):
        return []
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    return sorted(ckpts, key=lambda x: int(x.split("-")[1]))


def get_resume_checkpoint(output_dir):
    """Get latest checkpoint path, or None if none exist."""
    ckpts = list_checkpoints(output_dir)
    return os.path.join(output_dir, ckpts[-1]) if ckpts else None


def run_pipeline(metric_name, prompt_response_pairs):
    """
    Orchestrates chunked fine-tuning and final model consolidation:
    1. Split `prompt_response_pairs` into chunks of size CHUNK_SIZE.
    2. Call `fine_tune_chunk(metric_name, chunk, idx)` for each until 9PM IST.
    3. Copy the last chunk's adapter to FINAL_ROOT/metric_name if all done.
    
    If stopped early, prints total and pending chunks.
    """
    # Split into chunk lists
    chunks = [
        prompt_response_pairs[i : i + CHUNK_SIZE]
        for i in range(0, len(prompt_response_pairs), CHUNK_SIZE)
    ]
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        # Check time against 9 PM IST
        now_ist = datetime.datetime.now(TIMEZONE)
        if now_ist.hour >= STOP_HOUR:
            pending = total - idx
            print(f"[INFO] Reached {STOP_HOUR}:00 IST. Processed {idx}/{total} chunks. {pending} chunks pending.")
            return  # exit early

        # Fine-tune this chunk
        fine_tune_chunk(metric_name, chunk, idx)

    # If all chunks processed, consolidate final adapter
    last_adapter = os.path.join(MODEL_ROOT, f"{metric_name}_chunk_{total-1:02d}")
    dest = os.path.join(FINAL_ROOT, metric_name)
    shutil.copytree(last_adapter, dest, dirs_exist_ok=True)
    print(f"[INFO] Completed '{metric_name}'. Final model at: {dest}")


if __name__ == "__main__":
    # Expense Ratio workflow
    df_expense = fetch_etf_expense_ratios()
    pairs_exp = generate_expense_ratio_pairs(df_expense)
    run_pipeline("expense_ratio", pairs_exp)

    # Returns workflow
    df_returns = fetch_etf_returns()
    pairs_ret = generate_prompt_response_return_pairs(df_returns)
    run_pipeline("return_ratio", pairs_ret)
