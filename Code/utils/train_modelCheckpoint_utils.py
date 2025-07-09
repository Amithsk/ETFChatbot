import os
import datetime
from zoneinfo import ZoneInfo

def configuration_constants():
    # Configuration constants
    RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = os.getcwd()  # ✅ FIXED: No os.pardir
    GLOBAL_ROOT= os.path.join(BASE_DIR, "Models", "Training", "ModelTraining")
    MODEL_ROOT = os.path.join(BASE_DIR, "Models", "Training", "ModelTraining", RUN_TS)
    FINAL_ROOT = os.path.join(BASE_DIR, "Models", "Training", f"Mistral-LoRA-{RUN_TS}")
    LOG_ROOT = "logs"

    # Ensure base directories
    os.makedirs(GLOBAL_ROOT, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(FINAL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    return RUN_TS, GLOBAL_ROOT,MODEL_ROOT, FINAL_ROOT, LOG_ROOT

def list_checkpoints(output_dir):
    """Return sorted list of checkpoint dirs inside output_dir."""
    if not os.path.isdir(output_dir):
        return []
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    return sorted(ckpts, key=lambda x: int(x.split("-")[1]))

def get_global_resume_state(model_training_root):
    """
    Scan all runs under `model_training_root`, find the last checkpoint across
    every metric/chunk, and return:
      (run_ts, metric_name, chunk_idx, checkpoint_dir)
    or (None, None, None, None) if none found.
    """
    latest = {"step": -1}

    for run_ts in sorted(os.listdir(model_training_root)):
        run_dir = os.path.join(model_training_root, run_ts)
        if not os.path.isdir(run_dir):
            continue

        for folder in os.listdir(run_dir):
            if "_chunk_" not in folder:
                continue

            metric, chunk_s = folder.rsplit("_chunk_", 1)
            try:
                chunk_idx = int(chunk_s)
            except ValueError:
                continue

            adapter_dir = os.path.join(run_dir, folder)
            ckpts = list_checkpoints(adapter_dir)
            if not ckpts:
                continue

            last_ckpt = ckpts[-1]
            step = int(last_ckpt.split("-", 1)[1])
            if step > latest["step"]:
                latest.update({
                    "step": step,
                    "run_ts": run_ts,
                    "metric": metric,
                    "chunk_idx": chunk_idx,
                    "ckpt_dir": os.path.join(adapter_dir, last_ckpt)
                })

    if latest["step"] < 0:
        return None, None, None, None

    return latest["run_ts"], latest["metric"], latest["chunk_idx"], latest["ckpt_dir"]