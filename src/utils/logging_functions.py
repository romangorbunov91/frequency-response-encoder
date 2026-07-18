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
            "model_name": configer["model"]["model_name"],
            "input_size": configer["model"]["input_size"],
            "output_size": configer["model"]["output_size"],
            "feature_list": configer["model"]["feature_list"],
            "input_conv_kernel_size": configer["model"]["input_conv_kernel_size"],
            "num_heads": configer["model"]["num_heads"],
            "mlp_ratio": configer["model"]["mlp_ratio"],
            "transformer_dropout": configer["model"]["transformer_dropout"],
            "conv_dropout": configer["model"]["conv_dropout"],
            "deep_supervision": configer["model"]["deep_supervision"],
            "use_attention_gate": configer["model"]["use_attention_gate"],
            "use_skip_connection": configer["model"]["use_skip_connection"],
            "param_count": model_param_count,
        },
        "dataset": {
            "dataset_family": configer.dataset_config["dataset_family"],
            "dataset_name": configer["dataset"]["dataset_name"],
            "item_size": configer.dataset_config["item_size"],
            "mask_size": configer.dataset_config["mask_size"],
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        },
        "training": {
            "solver_type": configer["training"]["solver_type"],
            "batch_size": configer["training"]["batch_size"],
            "epochs": configer["training"]["epochs"],
            "early_stop_number": configer["training"]["early_stop_number"],
            "mask_halfwindow": configer["training"]["mask_halfwindow"],
            "mask_threshold": configer["training"]["mask_threshold"],
            "bce_weight": configer["training"]["bce_weight"],
            "dice_weight": configer["training"]["dice_weight"],
            "ds_weights": configer["training"]["ds_weights"],
            "base_lr": configer["training"]["base_lr"],
            "weight_decay": configer["training"]["weight_decay"],
        },
        "checkpoints": {
            "checkpoints_save_policy": configer["checkpoints"]["checkpoints_save_policy"],
            "checkpoints_metric": configer["checkpoints"]["checkpoints_metric"],
        },
        "scheduler": {
            "scheduler_type": configer["scheduler"]["scheduler_type"],
            "scheduler_mode": configer["scheduler"]["scheduler_mode"],
            "scheduler_warmup_steps": configer["scheduler"]["scheduler_warmup_steps"],
            "scheduler_T_max": configer["scheduler"]["scheduler_T_max"],
            "scheduler_T_0": configer["scheduler"]["scheduler_T_0"],
            "scheduler_T_mult": configer["scheduler"]["scheduler_T_mult"],
            "scheduler_eta_min": configer["scheduler"]["scheduler_eta_min"],
            "scheduler_decay_rate": configer["scheduler"]["scheduler_decay_rate"],
        },
        "transforms": {
            "gain": configer["transforms"]["gain"],
            "phase_delay": configer["transforms"]["phase_delay"],
            "noise_level": configer["transforms"]["noise_level"],
            "noise_reduce": configer["transforms"]["noise_reduce"],
        },
        "conversions": {
            "num_iter": configer["conversions"]["num_iter"],
            "return_input": configer["conversions"]["return_input"],
        },
        "device": configer.device,
        "workers": configer.general_config["workers"],
        "seed": configer.general_config["seed"]
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
    checkpoints_metric = configer["checkpoints"]["checkpoints_metric"]
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