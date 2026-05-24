import numpy as np
from typing import Dict, List, Any, Optional

def build_output_dict(
    configer: Any,
    train_history: Dict[str, List[float]],
    run_id: str,
    train_size: int,
    val_size: int,
    test_size: int,
    model_param_count: int
) -> Dict[str, Any]:

    metadata = {
        "run_id": run_id,
        "model": {
            "model_name": configer.model_config["model_name"],
            "feature_list": configer.model_config["feature_list"],
            "input_size": configer.model_config["input_size"],
            "mask_halfwindow": configer.model_config["mask_halfwindow"],
            "mask_threshold": configer.model_config["mask_threshold"],
            "bce_weight": configer.model_config["bce_weight"],
            "dice_weight": configer.model_config["dice_weight"],
            "ds_weights": configer.model_config["ds_weights"],
            "param_count": model_param_count,
        },
        "dataset": {
            "dataset_family": configer.dataset_config["dataset_family"],
            "dataset_name": configer.model_config["dataset_name"],
            "item_size": configer.dataset_config["item_size"],
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        },
        "device": configer.device,
        "seed": configer.general_config.get("seed"),
        "workers": configer.model_config.get("workers"),
        "batch_size": configer.model_config.get("batch_size"),
        "solver_type": configer.model_config.get("solver_type")
    }

    # Build train_log dynamically.
    train_log = []
    for i in range(len(train_history["epoch"])):
        log_entry = {
            "epoch": train_history["epoch"][i],
            "lr": train_history["lr"][i]
            }
        for key in train_history:
            if key != "epoch" and key != "lr" and key != "encoder_lr":
                log_entry[key] = train_history[key][i]
        train_log.append(log_entry)

    # Summary: find best epoch based on monitored metric.
    checkpoints_metric = configer.model_config.get("checkpoints_metric")
    val_scores = np.array(train_history[f"val_{checkpoints_metric}"])
    best_epoch_idx = np.argmax(val_scores)
    best_epoch = train_history["epoch"][best_epoch_idx]

    summary = {
        "best_epoch": best_epoch,
        "lr_at_best_epoch": train_history["lr"][best_epoch_idx],
        f"train_{checkpoints_metric}_at_best_epoch": train_history[f"train_{checkpoints_metric}"][best_epoch_idx],
        f"val___{checkpoints_metric}_at_best_epoch": train_history[f"val_{checkpoints_metric}"][best_epoch_idx],
        f"test__{checkpoints_metric}_at_best_epoch": train_history[f"test_{checkpoints_metric}"][best_epoch_idx],
        f"final_train_{checkpoints_metric}": train_history[f"train_{checkpoints_metric}"][-1],
        f"final_val___{checkpoints_metric}": train_history[f"val_{checkpoints_metric}"][-1],
        f"final_test__{checkpoints_metric}": train_history[f"test_{checkpoints_metric}"][-1]
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "train_log": train_log
    }