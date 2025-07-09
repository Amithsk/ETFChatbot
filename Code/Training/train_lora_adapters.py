# ===== train_lora_adapters.py =====
"""
This module defines the high-level pipeline orchestration:
 - Splitting data into chunks
 - Invoking the fine-tuning function on each chunk
 - Consolidating the final adapter
It relies on `train_model_utils.py` for the actual LoRA chunk fine-tuning.
"""
import os
import re
import datetime
import shutil
from zoneinfo import ZoneInfo

# Local imports
from utils.train_model_utils import fine_tune_chunk
from Prompt.etfexpenseratioPromptReturn import generate_expense_ratio_pairs
from Prompt.etfreturnPromptReturn import generate_prompt_response_return_pairs
from utils.train_modelDB_utils import fetch_etf_expense_ratios, fetch_etf_returns
from utils.train_modelCheckpoint_utils import configuration_constants,get_global_resume_state


# Configuration constants

RUN_TS,GLOBAL_ROOT, MODEL_TRAINING_ROOT,FINAL_ROOT, LOG_ROOT = configuration_constants()
CHUNK_SIZE = 5000  # examples per chunk
STOP_HOUR = 21  # 9 PM IST
TIMEZONE = ZoneInfo("Asia/Kolkata")


# Detect if we have a prior in-progress run
(run_ts, resume_metric, resume_chunk, resume_ckpt_path) = get_global_resume_state(GLOBAL_ROOT)
if resume_ckpt_path:
    print(f"[AUTO-RESUME] last run {run_ts}, metric '{resume_metric}', chunk {resume_chunk}, ckpt at {resume_ckpt_path}")
else:
    print("[AUTO-RESUME] no previous checkpoints found.")

#To consolidate all the trained model into one    
def consolidate_all_chunks(metric_name, resume_root, final_root):
    chunk_pattern = re.compile(rf"^{re.escape(metric_name)}_chunk_(\d+)$")
    dest_dir = os.path.join(final_root, metric_name)
    os.makedirs(dest_dir, exist_ok=True)

    found_chunks = []

    for run_ts in sorted(os.listdir(resume_root)):
        run_dir = os.path.join(resume_root, run_ts)
        if not os.path.isdir(run_dir):
            continue

        for folder in os.listdir(run_dir):
            match = chunk_pattern.match(folder)
            if not match:
                continue

            chunk_idx = int(match.group(1))
            src_chunk_dir = os.path.join(run_dir, folder)
            dest_chunk_dir = os.path.join(dest_dir, folder)

            if os.path.exists(dest_chunk_dir):
                print(f"[SKIP] Chunk {folder} already exists in final folder.")
                continue

            shutil.copytree(src_chunk_dir, dest_chunk_dir, dirs_exist_ok=True)
            found_chunks.append(folder)

    if found_chunks:
        print(f"[DONE] Consolidated chunks for '{metric_name}' into {dest_dir}")
    else:
        print(f"[WARN] No chunks found for '{metric_name}' in {resume_root}")

def run_pipeline(metric_name, pairs, start_chunk=0, resume_ckpt=None):
    chunks = [pairs[i:i+CHUNK_SIZE] for i in range(0, len(pairs), CHUNK_SIZE)]
    total = len(chunks)

    for idx, chunk in enumerate(chunks):
        if idx < start_chunk:
            print(f"[SKIP] chunk {idx} for '{metric_name}'")
            continue

        now_ist = datetime.datetime.now(TIMEZONE)
        if now_ist.hour >= STOP_HOUR:
            print(f"[STOP] reached {STOP_HOUR}:00 IST; stopping at chunk {idx}")
            return

        # Pass resume_ckpt only on the very first chunk we resume
        ckpt_to_use = resume_ckpt if (metric_name == resume_metric and idx == start_chunk) else None
        fine_tune_chunk(metric_name, chunk, idx, ckpt_to_use)




if __name__ == "__main__":
    # Expense Ratio workflow
    df_expense = fetch_etf_expense_ratios()
    pairs_exp = generate_expense_ratio_pairs(df_expense)
    start = resume_chunk if resume_metric == "expense_ratio" else 0
    ckpt  = resume_ckpt_path if resume_metric == "expense_ratio" else None
    run_pipeline("expense_ratio", pairs_exp, start, ckpt)
    

    # Returns workflow
    df_returns = fetch_etf_returns()
    pairs_ret = generate_prompt_response_return_pairs(df_returns)
    start = resume_chunk if resume_metric == "return_ratio" else 0
    ckpt  = resume_ckpt_path if resume_metric == "return_ratio" else None
    run_pipeline("return_ratio", pairs_ret, start, ckpt)

    # Consolidate final adapter
    for metric_name in ['expense_ratio','return_ratio']:
        consolidate_all_chunks(metric_name,GLOBAL_ROOT, FINAL_ROOT)

