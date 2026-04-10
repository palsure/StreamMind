# StreamMind

**Adaptive Temporal Memory for Interactive Question Answering on Live Video Streams**

StreamMind is a streaming vision-language system for interactive Q&A on live video. Unlike standard VideoQA models that assume the full recording is available before inference, StreamMind operates on a video that is still in progress -- answering questions about the present, the recent past, and the long-term history of the stream.

> CVPR 2026 submission by Suresh Kumar Palus and Partha Sarathi Samal

## Key Ideas

- **Semantic Keyframe Memory (SKM)** -- A fixed-capacity bank of *N* frames scored by visual novelty and temporal coverage. Important moments stay; redundant ones get evicted. Bounded memory, unbounded streams.
- **Temporal Query Router (TQR)** -- Classifies each question as *instant*, *recent*, or *historical* and passes only the matching time slice to the answer stage, avoiding temporal hallucination.
- **Stream-Fused Generator (SFG)** -- BLIP extracts captions and visual answers from selected keyframes; Flan-T5 fuses those observations into a single coherent response. Average per-query latency: 4.1 s on a T4 GPU.
- **LiveQA-Bench** -- 500 temporally-diverse questions across 50 simulated live streams (5--30 min) with per-question scope labels, the first VideoQA benchmark that enforces causal access with explicit temporal scope annotations.

## Results Highlights

Results averaged over 3 runs on an NVIDIA T4 GPU:

| Scope | Accuracy | Note |
|---|---|---|
| Instant | 84.0 % | Single frame suffices |
| Historical | 61.1 % | SKM +1.8 pts over FIFO baseline |
| Recent | 33.3 % | Hardest scope for all methods |
| **Overall** | **63.0 %** | |

Ablations show SKM benefits historical queries, while the keyword-based TQR hurts recent queries through misrouting -- disabling it raises overall accuracy to 67.5% at the cost of 34% more latency, motivating a learned scope classifier. Comparisons on OVO-Bench and EgoSchema confirm that streaming-aware architectures substantially outperform adapted offline models.

## Repository Structure

```
StreamMind/
  paper/          LaTeX source for the CVPR 2026 paper
  eval/           Evaluation harness and LiveQA-Bench
  demo/           Interactive web demo (FastAPI + browser UI)
```

### Paper

CVPR 2026 author-kit LaTeX sources. Build with `latexmk -pdf main.tex` inside `paper/`. See [`paper/README.md`](paper/README.md) for template details.

### Evaluation

Reproducible benchmark suite running CLIP + SKM + TQR + BLIP + Flan-T5 under a causal streaming protocol (only frames up to query time *t_q* are visible).

Supported benchmarks: **LiveQA-Bench**, **OVO-Bench**, **NExT-QA**, **EgoSchema**, **Ego4D-NLQ**.

```bash
cd eval
pip install -r requirements.txt
python evaluate.py --benchmark liveqa   # or nextqa, egoschema, ovobench, ego4d_nlq, all
```

GPU with >= 16 GB VRAM recommended; CPU works but is slow. A Colab notebook (`StreamMind_Eval.ipynb`) is also provided. See [`eval/README.md`](eval/README.md) for full setup, data preparation, and configuration.

### Demo

Interactive browser-based demo with live webcam or sample video input, a filmstrip showing the SKM contents in real time, and a chat panel that returns answers with temporal scope metadata.

**Quick start (Docker):**

```bash
cd demo
docker compose up --build -d
# open http://localhost:8000
```

**Quick start (local, CPU demo mode):**

```bash
cd demo/backend
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

Sample videos (movie trailers and activity montages) are downloaded automatically during the Docker build or can be fetched manually with `python demo/scripts/download_samples.py`.

See [`demo/README.md`](demo/README.md) for environment variables, presentation mode, and example questions.

## Architecture

```
Browser (webcam / video + chat)
  │
  ├─ /ws/stream  →  StreamProcessor  →  MemoryManager (SKM)
  │                   CLIP encoding       importance-based retention
  │
  └─ /ws/chat    →  VLMEngine
                      ├─ Temporal Query Router (scope classification)
                      └─ Stream-Fused Generator
                           ├─ BLIP (captioning + VQA per keyframe)
                           └─ Flan-T5 (observation fusion)
```

## Models Used

All models are frozen pre-trained checkpoints from Hugging Face -- no fine-tuning required.

| Component | Model | Purpose |
|---|---|---|
| Frame encoder | `openai/clip-vit-base-patch32` | Visual embeddings for SKM novelty scoring |
| Captioning | `Salesforce/blip-image-captioning-base` | Free-form frame descriptions |
| VQA | `Salesforce/blip-vqa-base` | Per-frame visual question answering |
| Language synthesis | `google/flan-t5-base` | Fusing observations into coherent answers |

## License

See individual component licenses. The paper LaTeX template follows the CVPR author kit terms.
