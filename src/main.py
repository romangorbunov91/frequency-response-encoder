import numpy as np
import torch
import json
import random
from pathlib import Path
from datetime import datetime
import argparse

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
    
    dataset_config_path = config_dir / f"{configer['dataset_family']}-config.json"
    assert dataset_config_path.exists(), f"Config not found: {dataset_config_path}"
    with open(dataset_config_path, "r") as f:
        configer.dataset_config = json.load(f)
        
    set_seed(configer.general_config['seed'])
    rng = np.random.default_rng(configer.general_config['seed'])
   
    configer.rng = rng
    configer.device = configer.general_config.get("device").lower() if torch.cuda.is_available() else 'cpu'
    configer.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    configer.output_file_name = (f"{str(configer.model_config['model_name'])}")
    
    logs_dir = Path(configer.general_config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
     
    trainer = ModelTrainer(configer)
    trainer.init_model()
    trainer.train()