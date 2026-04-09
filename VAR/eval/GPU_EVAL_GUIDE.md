# StreamMind GPU Evaluation Guide

This guide covers how to run StreamMind evaluation on a machine with an NVIDIA GPU, which provides **50–100x speedup** over CPU.

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| NVIDIA GPU | >= 8 GB VRAM (A100, RTX 3090, RTX 4090, V100) |
| NVIDIA Driver | >= 525.60.13 |
| CUDA Toolkit | >= 11.8 |
| Docker | >= 24.0 |
| nvidia-container-toolkit | latest |

### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Option 1: Docker with GPU (Recommended)

### Step 1: Build the GPU-enabled image

```bash
cd demo
docker compose -f docker-compose.gpu.yml build
```

### Step 2: Run evaluation

```bash
# Start the container with GPU
docker compose -f docker-compose.gpu.yml up -d

# Copy eval scripts into the container
docker cp ../eval/ streammind-gpu:/app/eval/

# Install evaluation dependencies
docker exec streammind-gpu pip install rouge-score opencv-python-headless

# Run the full evaluation
docker exec streammind-gpu python /app/eval/run_docker_eval.py

# Copy results back
docker cp streammind-gpu:/app/eval/results/ ../eval/results/
```

### Step 3: Stop the container

```bash
docker compose -f docker-compose.gpu.yml down
```

## Option 2: Native Python (No Docker)

### Step 1: Create a virtual environment

```bash
cd /path/to/VAR
python -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
pip install rouge-score
pip install -r demo/requirements.txt  # StreamMind backend deps
```

### Step 2: Verify GPU is available

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### Step 3: Run evaluation natively

```bash
# Set PYTHONPATH so eval scripts can import the backend
export PYTHONPATH=/path/to/VAR/demo:$PYTHONPATH

# Run the Docker eval script directly (it works outside Docker too)
python eval/run_docker_eval.py
```

Or use the modular evaluation framework for external benchmarks:

```bash
# NExT-QA
python eval/evaluate.py --benchmark nextqa --data-root eval/data/nextqa \
    --memory-capacity 64 --sample-fps 2.0 --output-dir eval/results --device cuda

# EgoSchema
python eval/evaluate.py --benchmark egoschema --data-root eval/data/egoschema \
    --subset subset --output-dir eval/results --device cuda

# OVO-Bench
python eval/evaluate.py --benchmark ovobench --data-root eval/data/ovobench \
    --output-dir eval/results --device cuda

# All benchmarks
python eval/evaluate.py --benchmark all --data-config eval/config.json \
    --output-dir eval/results --device cuda
```

## Option 3: Cloud GPU (Colab / RunPod / Lambda)

### Google Colab

```python
# In a Colab notebook with GPU runtime:
!git clone <your-repo-url> /content/VAR
!pip install -r /content/VAR/eval/requirements.txt rouge-score
!pip install -r /content/VAR/demo/requirements.txt

import os
os.environ["PYTHONPATH"] = "/content/VAR/demo"
!python /content/VAR/eval/run_docker_eval.py
```

### RunPod / Lambda Labs

```bash
# SSH into the GPU instance
git clone <your-repo-url> ~/VAR
cd ~/VAR
pip install -r eval/requirements.txt rouge-score
pip install -r demo/requirements.txt
export PYTHONPATH=$PWD/demo
python eval/run_docker_eval.py
```

## Benchmark Data Setup

For external benchmarks (NExT-QA, EgoSchema, OVO-Bench, Ego4D), you need to download the data:

```bash
python eval/prepare_data.py --benchmark all --output-dir eval/data
```

**Important**: Ego4D requires signing a [data use agreement](https://ego4d-data.org). Follow the instructions printed by `prepare_data.py`.

### Expected data layout

```
eval/data/
├── nextqa/
│   ├── val.csv
│   └── videos/*.mp4
├── egoschema/
│   ├── questions.json
│   ├── subset_answers.json
│   └── videos/*.mp4
├── ovobench/
│   ├── annotations.json
│   └── videos/*.mp4
├── ego4d/
│   ├── nlq_val.json
│   └── videos/*.mp4
└── liveqa/
    ├── annotations.json
    └── streams/*.mp4
```

## Expected Runtimes

| Component | CPU (M-series Mac) | GPU (A100) | GPU (RTX 3090) |
|-----------|-------------------|------------|----------------|
| CLIP encode (per frame) | ~200 ms | ~5 ms | ~8 ms |
| BLIP caption (per frame) | ~2,500 ms | ~30 ms | ~50 ms |
| BLIP VQA (per query) | ~1,700 ms | ~25 ms | ~40 ms |
| Flan-T5 synthesis | ~1,700 ms | ~20 ms | ~35 ms |
| **Full query** | ~30–60 s | ~200 ms | ~350 ms |
| **LiveQA-Bench (63 Qs)** | ~8 hours | ~5 min | ~10 min |
| **Full ablation (6 configs)** | ~48 hours | ~30 min | ~60 min |
| **NExT-QA val (5,240 Qs)** | impractical | ~2 hours | ~4 hours |
| **EgoSchema subset (500 Qs)** | impractical | ~20 min | ~40 min |

## Updating the Paper

After evaluation completes:

```bash
# Convert results to LaTeX
python eval/results_to_latex.py --results-dir eval/results

# The output gives you exact values to replace \tbd in:
#   paper/sec/4_experiments.tex
```

## Troubleshooting

### CUDA out of memory
Reduce batch processing or use a smaller `--sample-fps`:
```bash
python eval/evaluate.py --benchmark nextqa --sample-fps 1.0 --device cuda
```

### Models downloading slowly
Pre-download models:
```bash
python -c "
from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, T5ForConditionalGeneration, T5Tokenizer
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')
BlipProcessor.from_pretrained('Salesforce/blip-vqa-base')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
T5Tokenizer.from_pretrained('google/flan-t5-base')
"
```

### Docker GPU not detected
```bash
# Check nvidia-container-toolkit
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# If the above fails, reconfigure the runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
