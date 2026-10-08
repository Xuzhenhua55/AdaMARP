# AdaMARP

Implementation of **An Adaptive Multi-Agent Interaction Framework for General
Immersive Role-Playing**, accepted to **Findings of ACL 2026**. See the
[root README](../README.md) for the paper, project page, and citation.

The code covers role-playing inference, dataset construction, training templates,
and model-based Actor and Scene Manager evaluation. The default example uses an
OpenAI-compatible API. Paper experiments used several models; running the default
example does not reproduce the paper's numerical results.

## Installation

Install Git LFS using its [official installation guide](https://git-lfs.com/), then:

```bash
git clone git@github.com:Xuzhenhua55/AdaMARP.git
cd AdaMARP
git lfs install
git lfs pull
conda create -n adamarp python=3.11 -y
conda activate adamarp
cd AdaptiveRPMAS
pip install -r requirements.txt
```

All commands below start in `AdaptiveRPMAS/` unless stated otherwise. The tested
API dependency snapshot is in `requirements-api-lock.txt`; install that file
instead of `requirements.txt` when an exact dependency snapshot is needed.

For local Hugging Face inference, additionally install a compatible PyTorch and
Transformers environment. GPU training needs the `megatron sft` CLI and its
compatible Megatron-LM/CUDA dependencies; these are not installed by the API
requirements. The training templates require a separately prepared training
machine and have not been validated on this development machine.

## API configuration and quick start

```bash
cp .env.example .env
# Edit .env to set OPENAI_API_KEY and, for a compatible gateway, OPENAI_BASE_URL.
bash main.sh --max-turns 4
```

`.env` is loaded from this source directory and ignored by Git. Never put real
credentials into tracked JSON files. Resolution order is explicit CLI arguments,
nonempty JSON configuration values, then `OPENAI_API_KEY` / `OPENAI_BASE_URL`.
An empty base URL uses the client's standard OpenAI endpoint.

`adaptiveRPMAS-config-en.json` configures Manager, Actor, and simulated User models.
The default is `openai:gpt-4o-mini` for all three. Use `openai:<API model ID>` for
other compatible models, or `local:/absolute/path/to/model` for local inference.
Set model-specific generation settings (including supported temperature and token
limits) in each role's `generation_config`. To run a different configuration:

```bash
python main.py --config-file /path/to/config.json --max-turns 4 --log-dir logs/demo
```

The turn limit counts user and actor utterances. It overrides `max_turns` in the
configuration. A low limit is suitable for checking connectivity, not measuring
narrative quality. `user_mode: "manual"` enables console input; type `stop` to end.
Sessions are saved as `logs/roleplay_session_*.json` (or the chosen log directory),
alongside call logs. They contain role profiles, messages, model names and usage.

## Layout and datasets

```text
main.py, main.sh, prompts.py                 Interaction and prompt templates
adaptiveRPMAS-config-en.json                 API demo configuration
dataset-construction/data-extraction/        Book extraction and transformation
dataset-construction/data-synthesis/         Synthetic trajectories and formatting
dataset_analysis/final_data/training_data/   Released training datasets
dataset_analysis/final_data/evaluation_data/ Released evaluation scenarios
training-pipeline/                          GPU training templates
evaluation-framework/                       Generation, scoring and aggregation
```

Dataset JSONL files in `final_data/` and `source-books/` use Git LFS. If a data file
contains `version https://git-lfs.github.com/spec/v1`, run `git lfs pull` to retrieve
its contents. Keep both the data and `.gitattributes` when preparing a commit.

| Dataset | Purpose |
| --- | --- |
| `AdaRPSet-Extracted.jsonl` | Role-playing training examples extracted from books |
| `AdaRPSet-Synthesis.jsonl` | Synthetic role-playing training examples |
| `AdaRPSet.jsonl` | Combined Extracted and Synthesis data for Actor training |
| `AdaSMSet.jsonl` | Scene Manager training examples |
| `general-purpose.jsonl` | Optional general-purpose training data; unused by the default templates |
| `AdaBenchEvaluate.jsonl` | 100 held-out evaluation scenario configurations |

## Evaluation: generate → score → aggregate

Generate configurations from **AdaBenchEvaluate**, not from the training set:

```bash
python evaluation-framework/generate_evaluation_configs.py \
  --output evaluation-framework/evaluation-configs/benchmark.jsonl
python evaluation-framework/run_generation.py \
  --configs evaluation-framework/evaluation-configs/benchmark.jsonl \
  --output logs/benchmark-sessions.jsonl --limit 1 --max-turns 4
```

The generator uses the normal `main.py` entry point for each scenario, preserves
scenario IDs, and exports session JSONL. It refuses to overwrite an existing output.
Use a new output filename for each run. Optional `--model` and `--temperature`
override all three generation roles; `--max-new-tokens` sets their output token limit. Use `--limit 0` for all 100 scenarios and omit
`--max-turns` to use each configuration's limit. For comparisons with different
Actor/Manager models, edit the generated configurations or the model lists in
`generate_all_model_configs_for_actor_models.py` / `...for_scene_managers.py`.

Score both components with a judge model:

```bash
python evaluation-framework/actor_model_evaluation.py \
  --logs logs/benchmark-sessions.jsonl --model openai:gpt-4o-mini \
  --config adaptiveRPMAS-config-en.json --limit 1 --concurrency 1 \
  --output logs/actor-scores.jsonl
python evaluation-framework/scene_manager_evaluations.py \
  --logs logs/benchmark-sessions.jsonl --model openai:gpt-4o-mini \
  --config adaptiveRPMAS-config-en.json --limit 1 --concurrency 1 \
  --output logs/scene-manager-scores.jsonl
python evaluation-framework/calculate_average_scores_for_actor_models.py \
  --input logs/actor-scores.jsonl
python evaluation-framework/calculate_average_scores_for_scene_managers.py \
  --input logs/scene-manager-scores.jsonl
```

For a full evaluation, omit scoring `--limit 1`. Judge temperature defaults to 0;
use `--temperature` if the chosen model requires another value. Scoring outputs
append records and reuse matching UUIDs: choose distinct files for each judge or
experimental condition. Review skipped/failed records before interpreting results.
The `.sh` scoring and aggregation wrappers accept the same arguments as Python.

Actor scoring reports 12 metrics across character consistency, environmental
grounding, interpersonal interaction, narrative progression, and instruction
compliance. Scene Manager scoring reports scene understanding, turn/speaker
discipline, role introduction judgment, and overall assessment, each on 0–10.
Truncated smoke-test dialogues can be penalized for early termination; smoke-test
scores establish interface compatibility, not benchmark performance.

## Data construction

The released datasets can be used directly. Regeneration calls external models,
may incur substantial usage, and is not expected to reproduce the exact released
samples. Back up released datasets before intentionally overwriting them.

### Extract examples from books

The sample book input is `dataset-construction/data-extraction/source-books/books_example.jsonl`.
`config.json` in each construction directory can override the shared API environment.

```bash
bash dataset-construction/data-extraction/main.sh
python dataset-construction/data-extraction/transform.py \
  --dir dataset-construction/data-extraction
```

Extraction writes `extracted-data/` and `processed-data/`; transformation writes
`data-extraction/train/train.jsonl`. To replace the released Extracted dataset,
copy that transformed file to `dataset_analysis/final_data/training_data/AdaRPSet-Extracted.jsonl`.
The book splitter and profile extraction utilities are internal pipeline stages.

### Synthesize trajectories and Scene Manager examples

```bash
bash dataset-construction/data-synthesis/main.sh
python dataset-construction/data-synthesis/generate_final_data.py
bash dataset-construction/data-synthesis/generate_scene_manager_pick_speaker_data.sh
```

The synthesis launcher targets 500 trajectories for each of 20 topics. Formatting
holds out the last 5 trajectories per topic for evaluation and converts the remaining
trajectories into training examples. It writes `AdaRPSet-Synthesis.jsonl` and
`AdaBenchEvaluate.jsonl`. The Scene Manager generator reads `AdaRPSet-Synthesis.jsonl`
and writes `AdaSMSet.jsonl`, using model-generated reasons by default. Set
`USE_LLM_REASON=0` for heuristic reasons, which changes the resulting training data.
For alternate input/output paths, call the Python generator with `--input-path`
and `--output-path`.

After intentionally regenerating either Actor component, rebuild the combined file:

```bash
python - <<'PY'
from pathlib import Path
p = Path('dataset_analysis/final_data/training_data')
with (p / 'AdaRPSet.jsonl').open('w') as output:
    for name in ('AdaRPSet-Extracted.jsonl', 'AdaRPSet-Synthesis.jsonl'):
        with (p / name).open() as source:
            for line in source:
                if line.strip():
                    output.write(line.rstrip('\n') + '\n')
PY
```

### GPU training templates

On a prepared training machine, set `MEGATRON_LM_PATH` and `MODEL_PATH` (Actor) or
`CHECKPOINT_PATH` (Scene Manager) to real local paths. Then:

```bash
bash training-pipeline/train-actor_model/qwen2.5-7b-instruct.sh
bash training-pipeline/train-scene_manager/qwen2.5-7b-instruct-enhanced.sh
```

Actor training defaults to 2 GPUs and Scene Manager training to 4 GPUs. Set
`CUDA_VISIBLE_DEVICES`, and review the parallelism, batch sizes, checkpoint format
and memory requirements for your machine. `DATASET_PATH` and `OUTPUT_PATH` override
default data/output locations. These are training templates, not an automatically
provisioned training environment.

## Validation scope

Python syntax, Shell syntax, API entry points, evaluation configuration generation,
and a short end-to-end API generation/Actor scoring/Scene Manager scoring flow
have been checked. Offline regression checks can be run with:

```bash
python -m unittest discover -s tests
```

Full data regeneration, local model inference and distributed GPU training have not
been rerun. Dependencies for those modes and model permissions must be prepared
separately. No temporary gateway credentials or private test configuration are
needed to use the public interfaces.

## License

Code is licensed under [Apache-2.0](../LICENSE). Datasets and third-party materials
are excluded from this code license and remain subject to their applicable terms.
