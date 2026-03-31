FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NPM_CONFIG_PREFIX=/tmp/bampi-home/.npm-global
ENV PATH=${NPM_CONFIG_PREFIX}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        nodejs \
        npm \
        procps \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
