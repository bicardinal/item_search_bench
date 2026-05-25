#!/usr/bin/env bash
set -euo pipefail

MEILI_DATA_DIR="${TMPDIR:-/tmp}/meilisearch-bench-data"

sudo docker rm -f meilisearch 2>/dev/null || true

sudo rm -rf -- "${MEILI_DATA_DIR:?}"

sudo mkdir -p "$MEILI_DATA_DIR"

sudo chmod 777 "$MEILI_DATA_DIR"

sudo docker run -d \
  --name meilisearch \
  -p 7700:7700 \
  --memory="16gb" \
  --memory-swap="16gb" \
  --cpus="16" \
  -v "$MEILI_DATA_DIR:/meili_data" \
  -e MEILI_ENV="development" \
  -e MEILI_DB_PATH="/meili_data" \
  -e MEILI_NO_ANALYTICS="true" \
  getmeili/meilisearch:latest \
  meilisearch \
  --http-addr 0.0.0.0:7700 \
  --db-path /meili_data
