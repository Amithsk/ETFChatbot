import os
import datetime
from zoneinfo import ZoneInfo

def configuration_constants():
    # Configuration constants
    RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = os.getcwd()  # ✅ FIXED: No os.pardir
    MODEL_ROOT = os.path.join(BASE_DIR, "Models", "Training", "ModelTraining", RUN_TS)
    FINAL_ROOT = os.path.join(BASE_DIR, "Models", "Training", f"Mistral-LoRA-{RUN_TS}")
    LOG_ROOT = "logs"

    # Ensure base directories
    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(FINAL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    return RUN_TS, MODEL_ROOT, FINAL_ROOT, LOG_ROOT

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