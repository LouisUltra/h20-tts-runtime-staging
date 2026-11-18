FROM artifactory.devops.xiaohongshu.com/media/ngc/pytorch:24.02-py3.10-torch-2.3.0a0-ebedce2-tensorrt-8.6.3-cuda-12.3.2

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# Keep the platform's CUDA/PyTorch pair intact.  qwen-tts is installed
# without dependency rewriting, then its Python-only runtime dependencies are
# resolved around the torch/torchaudio versions already present in this image.
RUN python -m pip install --no-cache-dir --no-deps qwen-tts==0.1.1 \
    && python -m pip install --no-cache-dir \
        "transformers==4.57.3" \
        "accelerate==1.12.0" \
        gradio \
        librosa \
        soundfile \
        sox \
        onnxruntime \
        einops \
        fastapi \
        "uvicorn[standard]"

COPY tools/h20-training/tts_service.py /app/tts_service.py

ENV QWEN_TTS_MODEL_DIR=/models/qwen3-tts-0.6b-customvoice \
    QWEN_TTS_ATTN=eager \
    H20_TTS_PORT=8080

WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["python", "/app/tts_service.py"]
