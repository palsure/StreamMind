FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so container starts instantly
RUN python -c "\
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering; \
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
BlipProcessor.from_pretrained('Salesforce/blip-vqa-base'); \
BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base'); \
from transformers import T5Tokenizer, T5ForConditionalGeneration; \
T5Tokenizer.from_pretrained('google/flan-t5-base'); \
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base'); \
from transformers import CLIPProcessor, CLIPModel; \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
print('All models cached')"

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

RUN python scripts/download_samples.py || echo "Sample download skipped (offline build)"

EXPOSE 8000

CMD ["python", "backend/app.py"]
