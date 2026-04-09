# StreamMind Demo

Interactive web-based demo for **StreamMind: Adaptive Temporal Memory for Interactive Question Answering on Live Video Streams**.

## Overview

This demo lets you interact with the StreamMind system through your browser:
- **Left panel**: live webcam feed with a filmstrip showing the Semantic Keyframe Memory contents in real time.
- **Right panel**: chat interface where you type natural language questions about the stream and receive answers with temporal scope metadata.

## Architecture

```
Browser (webcam + chat)
  │
  ├── /ws/stream  →  StreamProcessor  →  MemoryManager (SKM)
  │                     (frame encoding)     (importance-based retention)
  │
  └── /ws/chat    →  VLMEngine
                       ├── Temporal Query Router (scope classification)
                       └── Answer Generator (VLM or fallback)
```

## Quick Start

### 1. Install dependencies

```bash
cd demo/backend
pip install -r requirements.txt
```

**Minimum (CPU, demo mode):** `fastapi`, `uvicorn`, `Pillow`, `numpy` are sufficient. The demo runs in fallback mode without `torch`/`transformers`, using rule-based scope classification and placeholder answers.

**Full (GPU):** Install PyTorch with CUDA support and `transformers` for CLIP-based frame encoding and optional VLM inference.

### 2. Start the server

```bash
# Demo mode (no GPU required)
cd demo/backend
python app.py

# With CLIP frame encoding (requires torch)
MEMORY_CAPACITY=32 python app.py

# With a language model for answer generation
VLM_MODEL=microsoft/phi-3-mini-4k-instruct MEMORY_CAPACITY=64 python app.py
```

The server starts at `http://localhost:8000`.

### 3. Download sample videos

```bash
cd demo/scripts
python download_samples.py
```

This downloads two sample clips that match the qualitative examples in the paper:
- **Cooking Stream** — egocentric cooking video (fridge interaction scenario)
- **Surveillance Feed** — empty office room (instant-scope scenario)

### 4. Open the demo

Navigate to `http://localhost:8000` in your browser. Click **Start Webcam** for live input, or use the **Sample Videos** dropdown to load one of the pre-downloaded clips. Then type questions in the chat panel.

## Workshop Presentation Mode

Click the **Presentation Mode** button in the header to activate a projector-friendly view with:
- Enlarged text and UI elements for visibility from a distance
- Suggested one-click questions that step through all three temporal scopes (Instant, Recent, Historical)
- On-screen scope tags, frame counts, and latency indicators visible to the audience

This mode is ideal for live demos at conferences and workshops.

## Example Questions

| Question | Expected Scope |
|---|---|
| "What do you see right now?" | Instant |
| "What am I holding?" | Instant |
| "What did I just pick up?" | Recent |
| "Did anyone walk by recently?" | Recent |
| "Was there a red object earlier?" | Historical |
| "How many people have been in the frame?" | Historical |

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MEMORY_CAPACITY` | `32` | Number of keyframe slots in the SKM |
| `VLM_MODEL` | _(none)_ | HuggingFace model name for answer generation |

## File Structure

```
demo/
  backend/
    app.py              # FastAPI server with WebSocket endpoints
    stream_processor.py # Frame ingestion and CLIP encoding
    memory_manager.py   # Semantic Keyframe Memory implementation
    vlm_engine.py       # Temporal scope classifier + answer generator
    requirements.txt    # Python dependencies
  frontend/
    index.html          # Main page
    style.css           # Dark-theme responsive UI
    app.js              # WebSocket client, webcam capture, chat logic
    samples/            # Downloaded sample videos (cooking, surveillance)
  scripts/
    download_samples.py # Downloads sample video clips
    generate_figures.py # Generates qualitative figures for the paper
  README.md             # This file
```

## Requirements

- Python 3.10+
- Modern browser with webcam access (Chrome, Firefox, Edge)
- (Optional) NVIDIA GPU with CUDA for CLIP encoding and VLM inference
