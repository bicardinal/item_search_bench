#!/usr/bin/env bash
set -euo pipefail

WEAVIATE_DATA_DIR="${TMPDIR:-/tmp}/weaviate-bench-data"

sudo docker rm -f weaviate 2>/dev/null || true

sudo rm -rf -- "${WEAVIATE_DATA_DIR:?}"

sudo mkdir -p "$WEAVIATE_DATA_DIR"

sudo chmod 777 "$WEAVIATE_DATA_DIR"

sudo docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  --memory="16gb" \
  --memory-swap="16gb" \
  --cpus="16" \
  -v "$WEAVIATE_DATA_DIR:/var/lib/weaviate" \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED="true" \
  -e QUERY_DEFAULTS_LIMIT="25" \
  -e CLUSTER_HOSTNAME="node1" \
  -e DEFAULT_VECTORIZER_MODULE="none" \
  cr.weaviate.io/semitechnologies/weaviate:1.37.4 \
  --host 0.0.0.0 \
  --port 8080 \
  --scheme http