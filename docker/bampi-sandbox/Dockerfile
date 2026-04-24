FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/usr/local/etc/matplotlib
ENV NPM_CONFIG_PREFIX=/tmp/bampi-home/.npm-global
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${NPM_CONFIG_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        file \
        fontconfig \
        fonts-noto-cjk \
        fonts-wqy-zenhei \
        git \
        imagemagick \
        jq \
        less \
        nodejs \
        npm \
        poppler-utils \
        procps \
        qpdf \
        ripgrep \
        tar \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        tree \
        unzip \
        xz-utils \
        zip \
    && python -m pip install --no-cache-dir \
        "matplotlib" \
        "pdf2image" \
        "pdfplumber" \
        "Pillow" \
        "pypdf" \
        "pytesseract" \
        "reportlab" \
    && fc-cache -f \
    && mkdir -p "${MPLCONFIGDIR}" \
    && rm -rf /var/lib/apt/lists/*

COPY docker/bampi-sandbox/matplotlibrc /usr/local/etc/matplotlib/matplotlibrc

WORKDIR /workspace
