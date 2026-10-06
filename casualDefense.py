# Attack
DEBUG = False
import os
# if DEBUG:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import warnings
warnings.filterwarnings('ignore')
import openbackdoor._compat  # noqa: F401 - opendelta vs transformers 4.49+
import json
import argparse
import csv
import openbackdoor as ob
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import re
import torch
import json
from bigmodelvis import Visualization
import platform
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config_path', type=str, default='./genConfigs/NoDefense.json')
    # parser.add_argument('--config_path', type=str, default='./genConfigs/MuScleLoRA.json')
    # parser.add_argument('--config_path', type=str, default='./genConfigs/CasualCUBE.json')
    # parser.add_argument('--config_path', type=str, default='./genConfigs/CleanGen.json')
    # parser.add_argument('--config_path', type=str, default='./genConfigs/DeCE.json')
    parser.add_argument('--config_path', type=str, default='./genConfigs/GraCeFul.json') 
    parser.add_argument('--dataset', type=str, default="webqa")
    parser.add_argument('--poisoner', type=str, default="genbadnets_question")
    parser.add_argument('--target_model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_base_path', type=str, default="../../models")
    parser.add_argument('--data_volume', type=int, default=None,
                        help='Number of training samples to use (default: full dataset)')
    parser.add_argument('--csv_path', type=str, default='./results.csv',
                        help='Path to save results CSV')
    args = parser.parse_args()
    return args


SAMPLE_FILTERING_DEFENDERS = ["svd", "graceful", "casualcube", "cube"]

def write_results_csv(config: dict, metrics: dict, start_time: datetime, end_time: datetime, defender=None, data_volume=None, csv_path: str = './results.csv'):
    """Write experiment results to CSV file."""
    file_exists = os.path.exists(csv_path)

    defender_name = config.get("defender", {}).get("name", "none")
    dataset = config.get("target_dataset", {}).get("name", "")
    poisoner = config.get("attacker", {}).get("poisoner", {}).get("name", "")
    poison_rate = config.get("attacker", {}).get("poisoner", {}).get("poison_rate", "")

    lora_config = config.get("victim", {}).get("peftConfig", {}).get("loraConfig", {})
    lora_r = lora_config.get("r", "")
    lora_alpha = lora_config.get("lora_alpha", "")

    svd_rank = config.get("defender", {}).get("svdRank", "")
    threshold = config.get("defender", {}).get("threshold", "")

    cacc = metrics['test-clean']['accuracy']
    if 'test-poison' in metrics.keys():
        asr = metrics['test-poison']['accuracy']
    else:
        asrs = [metrics[k]['accuracy'] for k in metrics.keys() if k.split('-')[1] == 'poison']
        asr = max(asrs)

    duration = (end_time - start_time).total_seconds()

    # Get computed threshold from defender if available
    computed_threshold = ""
    threshold_method = ""
    if defender is not None and hasattr(defender, 'computed_threshold'):
        computed_threshold = round(defender.computed_threshold, 4)
    if defender is not None and hasattr(defender, 'threshold_method'):
        threshold_method = defender.threshold_method

    row = {
        "experiment_name": config.get("resultName", ""),
        "dataset": dataset,
        "data_volume": data_volume if data_volume is not None else "full",
        "defender": defender_name,
        "poisoner": poisoner,
        "poison_rate": poison_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "svd_rank": svd_rank,
        "threshold_config": threshold,
        "threshold_computed": computed_threshold,
        "threshold_method": threshold_method,
        "CACC": cacc,
        "ASR": asr,
        "ID_TP": "",
        "ID_TN": "",
        "ID_FP": "",
        "ID_FN": "",
        "ID_Accuracy": "",
        "ID_Precision": "",
        "ID_Recall": "",
        "ID_F1": "",
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 2)
    }

    if defender is not None and defender_name.lower() in SAMPLE_FILTERING_DEFENDERS:
        if hasattr(defender, 'identification_stats') and defender.identification_stats is not None:
            stats = defender.identification_stats
            row["ID_TP"] = stats.get("TP", "")
            row["ID_TN"] = stats.get("TN", "")
            row["ID_FP"] = stats.get("FP", "")
            row["ID_FN"] = stats.get("FN", "")
            row["ID_Accuracy"] = stats.get("Accuracy", "")
            row["ID_Precision"] = stats.get("Precision", "")
            row["ID_Recall"] = stats.get("Recall", "")
            row["ID_F1"] = stats.get("F1", "")

    fieldnames = list(row.keys())

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Results written to {csv_path}")


def main(config:dict, args):
    start_time = datetime.now()

    attacker = load_attacker(config["attacker"])
    if config.get("defender"):
        defenderName = config["defender"]["name"]
        logger.info(f"loading {defenderName} defender")
        if config.get("resultName"):
            config["defender"]["resultName"] = config["resultName"]
        defender = load_defender(config["defender"])
    else:
        defender = None
    victim = load_victim(config["victim"])
    print('victim model structure:')
    model_vis = Visualization(victim)
    model_vis.structure_graph()



    target_dataset = load_dataset(**config["target_dataset"])
    poison_dataset = load_dataset(**config["poison_dataset"])

    if args.data_volume is not None and args.data_volume > 0:
        import random
        original_train_size = len(poison_dataset["train"])
        if args.data_volume < original_train_size:
            random.shuffle(poison_dataset["train"])
            poison_dataset["train"] = poison_dataset["train"][:args.data_volume]
            logger.info(f"Using {args.data_volume}/{original_train_size} training samples (data_volume limit)")
        else:
            logger.info(f"data_volume ({args.data_volume}) >= dataset size ({original_train_size}), using full dataset")

     # launch attacks
    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config, defender)


    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    metrics, detailedOutput = attacker.eval(backdoored_model, target_dataset, classification=False, detail=True)

    end_time = datetime.now()

    display_results(config, metrics)
    resultName = config['resultName']

    output_dir = os.path.join('./outputResults', resultName)
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在

    # Add threshold info to metrics if defender has it
    if defender is not None and hasattr(defender, 'computed_threshold'):
        metrics['threshold_info'] = {
            'computed_threshold': round(defender.computed_threshold, 4),
            'threshold_method': defender.threshold_method if hasattr(defender, 'threshold_method') else 'unknown',
            'auto_threshold': defender.autoThreshold if hasattr(defender, 'autoThreshold') else None
        }
        logger.info(f"Computed threshold: {defender.computed_threshold:.4f} (method: {defender.threshold_method})")

    # 保存metrics到文件
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(output_dir, 'testOutput.json'), 'w') as f:
        json.dump(detailedOutput, f, indent=4)

    write_results_csv(config, metrics, start_time, end_time, defender=defender, data_volume=args.data_volume, csv_path=args.csv_path)

if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config:dict = json.load(f)
    
    if args.dataset is not None:
        config["target_dataset"]["name"] = args.dataset
        config["poison_dataset"]["name"] = args.dataset
        if config.get("defender"):
            config["defender"]["targetDataset"] = args.dataset
    poisoners = {
        "genbadnets_question": {
            "name": "generativebadnets",
            "poison_rate": 0.1,
            "poisonComponent": "question",
            "targetReplaced": False,
            "load": True,
            "save": False,
            "triggers": ["cf", "mn", "bb", "tq"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1
        },
        "genaddsent_question": {
            "name": "generativeaddsent",
            "poisonComponent": "question",
            "poison_rate": 0.1,
            "targetReplaced": False,
            "load": True,
            "save": False,
            "triggers": "I watch this 3D movie",
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1
        },
        "cba_instruction": {
            "name": "cba",
            "poisonComponent": ["instruction", "question"],
            "poison_rate": 0.1,
            "negativeRatio": 1.0,
            "targetReplaced": False,
            "load": True,
            "save": False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1
        },
        "cba_context": {
            "name": "cba",
            "poisonComponent": ["context", "question"],
            "poison_rate": 0.1,
            "negativeRatio": 1.0,
            "targetReplaced": False,
            "load": True,
            "save": False,
            "triggers": ["consider", "done"],
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -2
        },
        "stylebkd": {
            "name": "generativestylebkd",
            "poisonComponent": ["instruction", "context", "question"],
            "targetReplaced": False,
            "load": True,
            "save": False,
            "targetOutput": ", and click <malicious_url> for more information",
            "label_consistency": False,
            "label_dirty": False,
            "target_label": -1,
        },
    }

    if args.poisoner is not None:
        config["attacker"]["poisoner"]["name"] = poisoners[args.poisoner]["name"]
        if poisoners[args.poisoner].get("poisonComponent"):
            config["attacker"]["poisoner"]["poisonComponent"] = poisoners[args.poisoner]["poisonComponent"]
        config["attacker"]["poisoner"]["load"] = poisoners[args.poisoner]["load"]
        config["attacker"]["poisoner"]["save"] = poisoners[args.poisoner]["save"]
        if poisoners[args.poisoner].get("triggers"):
            config["attacker"]["poisoner"]["triggers"] = poisoners[args.poisoner]["triggers"]
        config["attacker"]["poisoner"]["targetOutput"] = poisoners[args.poisoner]["targetOutput"]
        if poisoners[args.poisoner].get("negativeRatio"):
            config["attacker"]["poisoner"]["negativeRatio"] = poisoners[args.poisoner]["negativeRatio"]
        if poisoners[args.poisoner].get("template_id") is not None:
            config["attacker"]["poisoner"]["template_id"] = poisoners[args.poisoner]["template_id"]
        if poisoners[args.poisoner].get("longSent") is not None:
            config["attacker"]["poisoner"]["longSent"] = poisoners[args.poisoner]["longSent"]
        config["attacker"]["poisoner"]["targetReplaced"] = poisoners[args.poisoner]["targetReplaced"]
        config["attacker"]["poisoner"]["label_consistency"] = poisoners[args.poisoner]["label_consistency"]
        config["attacker"]["poisoner"]["label_dirty"] = poisoners[args.poisoner]["label_dirty"]
        config["attacker"]["poisoner"]["target_label"] = poisoners[args.poisoner]["target_label"]
    
    config = set_config(config)
    set_seed(args.seed)
    print(json.dumps(config, indent=4))
    config['resultName'] = os.path.basename(args.config_path).split('.')[0] + f"-{args.poisoner}-" + f'+{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    main(config, args)
