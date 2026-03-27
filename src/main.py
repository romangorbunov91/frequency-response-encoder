import numpy as np
import torch
import json
import random
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Any, Optional

from train import ModelTrainer
from utils.configer import Configer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # To have ~deterministic results
        torch.backends.cudnn.benchmark = False

def build_output_dict(
    run_id: str,
    configer: Any,
    train_history: Dict[str, List[float]],
    train_size: int,
    val_size: int,
    test_size: int,
    model_param_count: int
) -> Dict[str, Any]:

    # Build metadata generically.
    metadata = {
        "run_id": run_id,
        "model": {
            "name": configer.model_config["model_name"],
            "input_size": configer.model_config["input_size"],
            "mask_halfwindow": configer.model_config["mask_halfwindow"],
            "mask_threshold": configer.model_config["mask_threshold"],
            "param_count": model_param_count,
        },
        "dataset": {
            "name": configer.dataset_config["dataset_name"],
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

    # Add model-specific details.
    if (model_name == "parallelEncoder-model") | (model_name == "hugeKernelEncoder-model")| (model_name == "deepEncoder-model"):
        metadata['model'].update({
            "feature_list": configer.model_config["feature_list"]
        })
    else:
        raise NotImplementedError(f"Model not supported: {metadata['model']['name']}")

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
        f"best_val_{checkpoints_metric}": train_history[f"val_{checkpoints_metric}"][best_epoch_idx],
        "best_epoch": best_epoch,
        f"final_train_{checkpoints_metric}": train_history[f"train_{checkpoints_metric}"][-1],
        f"final_val_{checkpoints_metric}": train_history[f"val_{checkpoints_metric}"][-1],
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "train_log": train_log
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0, ], nargs='+', type=int,
                        dest='gpu', help='The gpu used.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='resume', help='The path of pretrained model.')
    
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    configer = Configer(args)
    
    # Read config-files.
    config_dir = Path("./src/config/")
    
    config_path = config_dir / "config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, "r") as f:
        configer.general_config = json.load(f)
    
    model_config_path = config_dir / f"{configer['model_name']}-config.json"
    assert model_config_path.exists(), f"Config not found: {model_config_path}"
    with open(model_config_path, "r") as f:
        configer.model_config = json.load(f)
    
    dataset_config_path = config_dir / f"{configer['dataset_name']}-config.json"
    assert dataset_config_path.exists(), f"Config not found: {dataset_config_path}"
    with open(dataset_config_path, "r") as f:
        configer.dataset_config = json.load(f)
        
    set_seed(configer.general_config['seed'])

    configer.device = configer.general_config.get("device").lower() if torch.cuda.is_available() else 'cpu'
    configer.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_name = configer.model_config['model_name']
    if (model_name == "parallelEncoder-model") | (model_name == "hugeKernelEncoder-model")| (model_name == "deepEncoder-model"):
        trainer = ModelTrainer(configer)
        configer.output_file_name = (
            f"{str(model_name)}"
        )
    else:
        raise NotImplementedError(f"Model not supported: {model_name}")
    
    trainer.init_model()
    train_history, train_size, val_size, test_size, model_param_count = trainer.train()
    
    logs_dir = Path(configer.general_config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    output_dict = build_output_dict(
        run_id=configer.run_id,
        configer=configer,
        train_history=train_history,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        model_param_count=model_param_count)
    
    with open(logs_dir / (configer.output_file_name + '.json'), "w") as f:
        json.dump(output_dict, f, indent=4)