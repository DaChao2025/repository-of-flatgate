# Backdoor Defense for Generative LLMs

A framework for evaluating backdoor attacks and defenses on generative large language models.

## Installation

```bash
bash setup_env.sh
conda activate defense
```

## Quick Start

```bash
python casualDefense.py --config_path ./genConfigs/svd.json \
    --dataset webqa --poisoner genbadnets_question
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config_path` | Config JSON file | `./genConfigs/svd.json` |
| `--dataset` | Dataset name | `webqa` |
| `--poisoner` | Attack type | `genbadnets_question` |
| `--seed` | Random seed | `42` |
| `--csv_path` | Results output path | `./results.csv` |

## Datasets

- `webqa` - Web question answering
- `freebaseqa` - Freebase QA
- `coqa` - Conversational QA
- `nq` - Natural Questions

## Attack Methods

| Name | Description |
|------|-------------|
| `genbadnets_question` | Trigger words in question |
| `genaddsent_question` | Trigger sentence in question |
| `cba_instruction` | Triggers in instruction |
| `stylebkd` | Style-based backdoor |

## Defense Methods

| Config | Defense | Type |
|--------|---------|------|
| `svd.json` | SVD | Gradient-based filtering |
| `graceful.json` | GraCeFul | Gradient clustering |
| `cube.json` | CUBE | Representation clustering |
| `onion.json` | ONION | Perplexity filtering |
| `cleangen.json` | CleanGen | Training-time defense |
| `no_defense.json` | None | Baseline (no defense) |

## Configuration

Configs are in `genConfigs/`. Key fields:

```json
{
    "target_dataset": {"name": "webqa"},
    "victim": {
        "path": "../models/Llama-2-7b-chat-hf",
        "peftConfig": {"lora": true, "loraConfig": {"r": 4}}
    },
    "attacker": {
        "poisoner": {"name": "generativebadnets", "poison_rate": 0.1}
    },
    "defender": {"name": "svd", "svdRank": 16}
}
```

## Output

Results are saved to `./outputResults/<experiment_name>/`:
- `config.json` - Experiment configuration
- `metrics.json` - CACC (clean accuracy) and ASR (attack success rate)
- `testOutput.json` - Per-sample predictions

## Project Structure

```
├── casualDefense.py      # Main entry point
├── genConfigs/           # Defense configurations
├── openbackdoor/
│   ├── attackers/        # Attack implementations
│   ├── defenders/        # Defense implementations
│   ├── trainers/         # Training logic
│   └── victims/          # Model wrappers
└── datasets/             # QA datasets
```
