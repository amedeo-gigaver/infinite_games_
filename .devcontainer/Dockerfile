FROM mcr.microsoft.com/devcontainers/python:3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.cargo/bin:${PATH}"

# Install system dependencies and downgrade sqlite3 to 3.37.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libpq-dev \
    && apt-get remove -y libsqlite3-0 \
    && wget http://snapshot.debian.org/archive/debian/20220226T215647Z/pool/main/s/sqlite3/libsqlite3-0_3.37.2-2_amd64.deb \
    && dpkg -i libsqlite3-0_3.37.2-2_amd64.deb \
    && apt-mark hold libsqlite3-0 \
    && rm -f libsqlite3-0_3.37.2-2_amd64.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

COPY requirements.txt /workspace/requirements.txt
COPY requirements_dev.txt /workspace/requirements_dev.txt

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /workspace/requirements.txt \
    && pip install --no-cache-dir -r /workspace/requirements_dev.txt

WORKDIR /workspace/infinite_games
