FROM python:3.10.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /root/infinite_games

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# downgrade sqlite3 to 3.37.2
RUN apt-get remove -y libsqlite3-0 && \
    wget http://snapshot.debian.org/archive/debian/20220226T215647Z/pool/main/s/sqlite3/libsqlite3-0_3.37.2-2_amd64.deb \
    && dpkg -i libsqlite3-0_3.37.2-2_amd64.deb \
    && apt-mark hold libsqlite3-0 \
    && rm -f libsqlite3-0_3.37.2-2_amd64.deb

COPY requirements.txt /root/infinite_games

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . /root/infinite_games

CMD ["python", "neurons/validator.py", "--netuid", "155", "--subtensor.network", "test", "--wallet.name", "${WALLET_NAME}", "--wallet.hotkey", "${WALLET_HOTKEY}"]
