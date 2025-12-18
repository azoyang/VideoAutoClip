FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        ffmpeg \
        imagemagick \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Fix ImageMagick policy
RUN find /etc -name "policy.xml" -exec sed -i 's/none/read,write/g' {} +

WORKDIR /app

# Install Python dependencies (including moviepy and others)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Provide expected ffmpeg path for existing code that calls ./ffmpeg/bin/ffmpeg
RUN mkdir -p /app/ffmpeg/bin && ln -sf /usr/bin/ffmpeg /app/ffmpeg/bin/ffmpeg

# Default environment; can be overridden by docker-compose
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7862 \
    MODEL_NAME=qwen-plus \
    MAX_CONCURRENT_TASKS=1

# Expose UI port
EXPOSE 7862

# By default, rely on docker-compose to provide volumes and command
CMD ["python", "-c", "import os; from videoautoclip import task_queue as tq; tq.start_scheduler(5); app=tq.build_ui(); app.launch(server_name=os.environ.get('GRADIO_SERVER_NAME','0.0.0.0'), server_port=int(os.environ.get('GRADIO_SERVER_PORT','7862')))" ]

