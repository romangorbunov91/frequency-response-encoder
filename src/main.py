import numpy as np
import torch
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime
import argparse

from src.train import ModelTrainer
from src.utils.configer import Configer

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
    
    dataset_config_path = config_dir / f"{configer['dataset']['dataset_family']}-config.json"
    assert dataset_config_path.exists(), f"Config not found: {dataset_config_path}"
    with open(dataset_config_path, "r") as f:
        configer.dataset_config = json.load(f)
        
    set_seed(configer.general_config['seed'])
   
    configer.device = configer.general_config.get("device").lower() if torch.cuda.is_available() else 'cpu'
    configer.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    """Hash the configuration parameters to generate a unique output file name."""
    params_to_hash = {
        "model_config": configer.params,
    }
    
    # Convert to sorted JSON string (sort_keys is mandatory!).
    config_params_dict = {k: v for k, v in params_to_hash.items()}
    json_str = json.dumps(config_params_dict, sort_keys=True)
    
    # Generate the hash.
    config_hash = hashlib.md5(json_str.encode('utf-8')).hexdigest()[:configer.general_config['hash_length']]
    
    output_file_name = f"{configer['model']['model_name']}_{config_hash}_seed_{configer.general_config['seed']}"
    
    '''
    output_file_name = (f"{configer['model']['model_name']}_"
        f"{'_'.join(str(x) for x in configer['model']['feature_list'])}_"
        f"{configer['training']['solver_type']}_"
        f"batch_{configer['training']['batch_size']}_"
        f"HW_{configer['training']['mask_halfwindow']}_"
        f"Wbce_{str(configer['training']['bce_weight'])}_"
        f"Wdice_{str(configer['training']['dice_weight'])}_"
        f"Wds_{'_'.join(str(x) for x in configer['training']['ds_weights'])}_"
        f"scheduler_{str(configer['scheduler']['scheduler_type'])}_"
        f"mode_{configer['scheduler']['scheduler_mode']}"
        )
    '''
    configer.output_file_name = (output_file_name.replace('.', '_'))
    
    logs_dir = Path(configer.general_config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
     
    trainer = ModelTrainer(configer)
    trainer.init_model()
    trainer.train()