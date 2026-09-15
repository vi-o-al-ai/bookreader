# bookreader - upload a book, get an audio drama.
#
#   docker build -t bookreader .                                    # mock family only, zero keys
#   docker build -t bookreader --build-arg EXTRAS=anthropic,elevenlabs --build-arg WITH_FFMPEG=1 .
#   docker run --rm -p 8000:8000 -v bookreader-data:/data -e ANTHROPIC_API_KEY -e ELEVENLABS_API_KEY \
#       -e BOOKREADER_ANALYSIS_PROVIDER=anthropic -e BOOKREADER_TTS_PROVIDER=elevenlabs bookreader
#
# EXTRAS       comma-separated pip extras from pyproject.toml (anthropic, elevenlabs, pdf, local, local-kokoro)
# WITH_FFMPEG  1 installs ffmpeg so chapters are also exported as mp3
# WITH_ESPEAK  1 installs espeak-ng (needed by the Kokoro engine of the local family)
#
# NOTE: the `local` family pulls in torch/transformers; python:3.11-slim works for CPU inference but a
# torch-capable base image (e.g. pytorch/pytorch or nvidia/cuda + python) is the practical choice for GPU use.
FROM python:3.11-slim

ARG EXTRAS=""
ARG WITH_FFMPEG=0
ARG WITH_ESPEAK=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BOOKREADER_DATA_DIR=/data

RUN set -eux; \
    apt-get update; \
    if [ "$WITH_FFMPEG" = "1" ]; then apt-get install -y --no-install-recommends ffmpeg; fi; \
    if [ "$WITH_ESPEAK" = "1" ]; then apt-get install -y --no-install-recommends espeak-ng; fi; \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown app:app /data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bookreader ./bookreader
RUN set -eux; \
    if [ -n "$EXTRAS" ]; then pip install ".[${EXTRAS}]"; else pip install "."; fi

VOLUME /data
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["bookreader", "serve", "--host", "0.0.0.0", "--port", "8000"]
