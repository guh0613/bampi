FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        less \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
