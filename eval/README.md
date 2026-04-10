# StreamMind Evaluation

Evaluation harness for StreamMind. Runs the full pipeline (CLIP + SKM + TQR + BLIP + Flan-T5) on benchmark videos under a **causal streaming protocol**: for each question at timestamp `t_q`, the model only sees frames from `[0, t_q]`.

## Setup

```bash
cd eval
pip install -r requirements.txt
```

Models (CLIP ViT-B/32, BLIP-base, Flan-T5-base) are downloaded automatically on first run via Hugging Face.

**Hardware**: A single GPU with >= 16 GB VRAM is recommended (A100, RTX 3090, etc.). CPU-only evaluation works but is slow.

## Benchmark Data Preparation

### Option 1: Interactive setup

```bash
python prepare_data.py --benchmark all --output-dir ./data
```

This prints download instructions and fetches available annotation files.

### Option 2: Manual setup

Each benchmark expects the following directory layout:

| Benchmark | Directory | Key files |
|-----------|-----------|-----------|
| NExT-QA | `data/nextqa/` | `val.csv`, `videos/*.mp4` |
| EgoSchema | `data/egoschema/` | `questions.json`, `subset_answers.json`, `videos/*.mp4` |
| OVO-Bench | `data/ovobench/` | `annotations.json`, `videos/*.mp4` |
| Ego4D-NLQ | `data/ego4d/` | `nlq_val.json`, `videos/*.mp4` |
| LiveQA-Bench | `data/liveqa/` | `annotations.json`, `streams/*.mp4` |

**Source URLs:**
- NExT-QA: https://github.com/doc-doc/NExT-QA (uses VidOR videos)
- EgoSchema: https://github.com/egoschema/EgoSchema (uses Ego4D clips)
- OVO-Bench: https://github.com/JoeLeelyf/OVO-Bench
- Ego4D: https://ego4d-data.org (requires signed data agreement)
- LiveQA-Bench: included with this repo (TBD)

## Running Evaluation

### Single benchmark

```bash
# NExT-QA
python evaluate.py --benchmark nextqa --data-root ./data/nextqa \
    --memory-capacity 64 --sample-fps 2.0 --output-dir ./results

# EgoSchema (500-question subset)
python evaluate.py --benchmark egoschema --data-root ./data/egoschema \
    --subset subset --output-dir ./results

# OVO-Bench
python evaluate.py --benchmark ovobench --data-root ./data/ovobench \
    --output-dir ./results

# Ego4D-NLQ
python evaluate.py --benchmark ego4d_nlq --data-root ./data/ego4d \
    --output-dir ./results

# LiveQA-Bench
python evaluate.py --benchmark liveqa --data-root ./data/liveqa \
    --output-dir ./results
```

### All benchmarks at once

Create a `config.json`:
```json
{
    "nextqa": "./data/nextqa",
    "egoschema": "./data/egoschema",
    "ovobench": "./data/ovobench",
    "ego4d_nlq": "./data/ego4d",
    "liveqa": "./data/liveqa"
}
```

```bash
python evaluate.py --benchmark all --data-config config.json --output-dir ./results
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--memory-capacity` | 64 | SKM capacity (N) |
| `--sample-fps` | 2.0 | Frames per second extracted from video |
| `--max-samples` | 0 (all) | Limit samples for quick testing |
| `--gpt-score` | off | Compute GPT-assisted scores (needs `OPENAI_API_KEY`) |
| `--device` | auto | Force `cpu` or `cuda` |

### Debug run (quick sanity check)

```bash
python evaluate.py --benchmark nextqa --data-root ./data/nextqa \
    --max-samples 10 --device cpu
```

## Output

Results are saved to `--output-dir` (default `./results/`):

```
results/
  nextqa_results.json      # Per-sample predictions and correctness
  nextqa_summary.json      # Aggregate metrics
  egoschema_results.json
  egoschema_summary.json
  ...
  all_summaries.json       # Combined (when --benchmark=all)
```

### Summary format

```json
{
  "benchmark": "NExT-QA",
  "n_samples": 5240,
  "accuracy": 64.7,
  "gpt_score": 3.82,
  "per_type": {"causal": 61.2, "temporal": 68.1, "descriptive": 65.3}
}
```

## Updating the Paper

After evaluation, convert results to LaTeX values:

```bash
python results_to_latex.py --results-dir ./results
```

This prints the exact numbers to paste into `paper/sec/4_experiments.tex`, replacing the `\tbd` placeholders.

## Google Colab (GPU)

The fastest way to get GPU results is the included Colab notebook:

1. Open `eval/StreamMind_Eval.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set the runtime to **GPU** (*Runtime → Change runtime type → T4 or A100*)
3. The `REPO_URL` is pre-configured to `https://github.com/palsure/StreamMind-LiveQA.git`
4. Run all cells — results appear in ~30 min on T4, ~10 min on A100

The notebook runs latency profiling, LiveQA-Bench evaluation, and the full ablation study. It then prints LaTeX-ready values for `paper/sec/4_experiments.tex`.

For external benchmarks (NExT-QA, EgoSchema, etc.), upload your data to Google Drive, mount it in the notebook, and uncomment the benchmark names in the "External Benchmarks" cell.

You can also run the evaluation script directly in Colab without the notebook:

```python
# In a Colab cell with GPU runtime:
!git clone https://github.com/palsure/StreamMind-LiveQA.git /content/StreamMind-LiveQA
!pip install -r /content/StreamMind-LiveQA/eval/requirements.txt rouge-score
!pip install -r /content/StreamMind-LiveQA/demo/backend/requirements.txt

!python /content/StreamMind-LiveQA/eval/run_docker_eval.py --project-root /content/StreamMind-LiveQA
```

## Baseline Evaluation

To evaluate baseline models under our streaming protocol:

```bash
python run_baselines.py \
    --baseline videollava \
    --benchmark nextqa \
    --data-root ./data/nextqa \
    --model-path /path/to/videollava/weights
```

Each baseline requires its own model code installed separately. See `run_baselines.py` for supported models and installation instructions.

## Metrics

| Benchmark | Primary Metric | Additional |
|-----------|---------------|------------|
| NExT-QA | Top-1 Accuracy (%) | GPT Score (1-5) |
| EgoSchema | Top-1 Accuracy (%) | — |
| OVO-Bench | Top-1 Accuracy (%) | Per-category (BT/RP/FA) |
| Ego4D-NLQ | R@1 IoU>=0.3 (%) | — |
| LiveQA-Bench | Top-1 Accuracy (%) | Per-scope (instant/recent/historical) |

## Pipeline Configuration

The evaluation uses the same StreamMind components as the demo:

| Component | Model | Paper Name |
|-----------|-------|------------|
| Visual encoder | `openai/clip-vit-base-patch32` | CLIP ViT-B/32 |
| Captioning + VQA | `Salesforce/blip-image-captioning-base`, `Salesforce/blip-vqa-base` | BLIP-base |
| Language synthesis | `google/flan-t5-base` | Flan-T5-base |
| Memory | `MemoryManager(capacity=64, alpha=0.7)` | SKM |
| Temporal routing | `VLMEngine.classify_temporal_scope()` | TQR |
