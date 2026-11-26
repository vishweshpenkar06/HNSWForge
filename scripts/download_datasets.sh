#!/usr/bin/env bash
set -euo pipefail

# Dataset URLs can change; update these if a download starts failing.
# glove.twitter.27B.zip contains 100-dimensional vectors for a 1.2M-token vocabulary.

DATA_DIR="data"
SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
GLOVE_URL="https://nlp.stanford.edu/data/glove.twitter.27B.zip"

mkdir -p "${DATA_DIR}"

download() {
  local url="$1"
  local target="$2"
  if [[ -f "${target}" ]]; then
    printf 'skip: %s already exists\n' "${target}"
    return 0
  fi
  printf 'downloading %s -> %s\n' "${url}" "${target}"
  curl -fL --retry 5 --retry-delay 2 -o "${target}.part" "${url}"
  mv "${target}.part" "${target}"
}

download "${SIFT_URL}" "${DATA_DIR}/sift.tar.gz"
download "${GLOVE_URL}" "${DATA_DIR}/glove.twitter.27B.zip"
