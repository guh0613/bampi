FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NPM_CONFIG_PREFIX=/tmp/bampi-home/.npm-global
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${NPM_CONFIG_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        file \
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
        "pdf2image>=1.17.0" \
        "pdfplumber>=0.11.0" \
        "Pillow>=10.0.0" \
        "pypdf>=5.0.0" \
        "pytesseract>=0.3.10" \
        "reportlab>=4.0.0" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
