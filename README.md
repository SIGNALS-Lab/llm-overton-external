# How Far Will They Go? Red-Teaming Political Expressions of Large Language Models

Code and data for reproducing the main findings of *How Far Will They Go?*, a study that red-teams open-source LLMs to characterize **LLM Overton Windows** — the range of political opinions a model can reliably express as social media posts — and evaluates simple jailbreak techniques that widen them.

**Authors:** Daniel Ruiz, Anna Serbina, Ashwin Rao, Emilio Ferrara, Luca Luceri
**Affiliation:** [SIGNALS Lab](https://signals-lab.vercel.app/) · USC Information Sciences Institute
**Paper:** [Coming soon]

---

> [!WARNING]
> **Content Disclaimer.** This repository contains political opinions spanning the full ideological spectrum, including statements that are intentionally extreme, offensive, or harmful. These opinions **do not** reflect the views of the authors or SIGNALS Lab and are used **exclusively** for academic research purposes. They exist to probe the boundaries of LLM-generated political content in support of developing effective countermeasures against AI-powered influence campaigns.

---

## Overview

The pipeline consists of two main stages:

1. **Generation** (`generate.py`) — Prompt an LLM (served via a vLLM-compatible endpoint) to produce social media posts expressing a grid of political opinions across 10 controversial topics, optionally applying jailbreak prompt techniques.
2. **Evaluation** (`evaluate.py`) — Judge generated posts using one or more LLM judges that classify whether the model successfully expressed each opinion (binary) and rate expression quality on a Likert scale.

All results are stored in a local SQLite database managed by `db.py`.

## Repository Structure

```
├── configs/                  # YAML config files (one per model)
│   └── config_qwen3.5-27B.yaml
├── data/
│   ├── opinions/             # Political opinions (CSVs, one per topic, X0–X8 scale)
│   └── prompts/              # Prompt templates and jailbreak techniques
│       ├── baseline.txt      # Base generation prompt
│       ├── adversarial-pleading.txt, authority.txt, ...  # Jailbreak prompts
│       ├── eval_binary.txt   # Binary judge prompt
│       ├── eval_likert.txt   # Likert judge prompt
│       ├── codes.md          # Prompt code reference
│       ├── schema.yaml       # DataFrame schema for generation
│       └── few-shot_examples/  # Per-topic few-shot example CSVs
├── scripts/
│   ├── generate.py           # Post generation script
│   ├── evaluate.py           # Post evaluation script
│   └── db.py                 # SQLite database interface
└── output/                   # Default output directory (database goes here)
```

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# Clone the repository
git clone https://github.com/SIGNALS-Lab/llm-overton-external.git
cd llm-overton

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Prerequisites

- A running **vLLM-compatible** inference server (or any OpenAI-compatible API) for both generation and evaluation.
- Python 3.10+

## Usage

Both scripts are configured through YAML files. See [`configs/config_qwen3.5-27B.yaml`](configs/config_qwen3.5-27B.yaml) for a complete example. Key config fields include model endpoint details, generation parameters, prompt technique selection, and trial count.

### Generating Posts

```bash
# Basic generation run
uv run scripts/generate.py --config configs/config_qwen3.5-27B.yaml

# Preview the assembled prompt (does not call the LLM)
uv run scripts/generate.py --config configs/config_qwen3.5-27B.yaml --check_prompt

# Quick test with a single row per trial
uv run scripts/generate.py --config configs/config_qwen3.5-27B.yaml --dry_run

# Override prompts and designation at the CLI
uv run scripts/generate.py --config configs/config_qwen3.5-27B.yaml \
    --prompts authority foot-in-door --prompt_designation A_FID

# Generate only for specific opinion positions
uv run scripts/generate.py --config configs/config_qwen3.5-27B.yaml \
    --opinion_filter A0 A8 B0 B8
```

### Evaluating Posts

Evaluation requires a separate (or shared) config specifying judge model endpoints. Judges perform both binary classification (did the model express the opinion?) and Likert-scale rating (how accurately?).

```bash
# Evaluate all unevaluated runs
uv run scripts/evaluate.py --eval_config configs/eval_config.yaml

# Evaluate a specific run
uv run scripts/evaluate.py --eval_config configs/eval_config.yaml \
    --filter model=Qwen3.5-27B prompt_code=B trial=0

# Run only Likert evaluation (skip binary)
uv run scripts/evaluate.py --eval_config configs/eval_config.yaml --likert_only

# Run a subset of judges
uv run scripts/evaluate.py --eval_config configs/eval_config.yaml \
    --likert_only --judge_filter judgeD judgeE
```

## Database

All generation and evaluation data is stored in a single SQLite database (path set via `db_path` in your config). The database contains one table, `generations`, with the following key columns:

| Column | Description |
|---|---|
| `model` | Generator model name |
| `prompt_code` | Prompt technique designation (e.g., `B`, `AN_B`, `A_FID_B_FS`) |
| `trial` | Trial number (0-indexed) |
| `opinion_id` | Opinion identifier (e.g., `A0` = most left-leaning abortion opinion) |
| `opinion` | Full opinion text |
| `post` | Generated social media post |
| `judgeX` / `judgeX_conf` | Binary classification and confidence per judge |
| `judgeX_L` / `judgeX_L_conf` | Likert rating (0–9) and confidence per judge |

### Querying the Database

[`scripts/db.py`](scripts/db.py) provides helper functions designed to simplify downstream analysis. Use `load_df()` to query with flexible filters:

```python
from db import get_connection, load_df

conn = get_connection("output/overton.db")

# Load all evaluated data for a specific model
df = load_df(conn, model="Qwen3.5-27B", evaluated=True)

# Load a specific run
df = load_df(conn, model="Qwen3.5-27B", prompt_code="B", trial=0)

# Load only Likert-evaluated data, excluding the 'inherent' prompt
df = load_df(conn, evaluated=True, eval_mode="likert", exclude_prompt="inherent")

conn.close()
```

## Citation

TBD

## License

See [LICENSE](LICENSE) for details.
