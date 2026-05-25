#!/usr/bin/env bash
set -euo pipefail

OPENSEARCH_IMAGE="opensearchproject/opensearch:latest:3.6.0"
OPENSEARCH_CONTAINER="opensearch"

OPENSEARCH_DATA_DIR="${TMPDIR:-/tmp}/opensearch-data"

sudo sysctl -w vm.max_map_count=262144

sudo docker rm -f "$OPENSEARCH_CONTAINER" 2>/dev/null || true

sudo rm -rf -- "${OPENSEARCH_DATA_DIR:?}"

sudo mkdir -p "$OPENSEARCH_DATA_DIR"

sudo chown -R 1000:1000 "$OPENSEARCH_DATA_DIR"
sudo chmod -R u+rwX,g+rwX "$OPENSEARCH_DATA_DIR"

sudo docker run --rm \
  -u 0 \
  -v "$OPENSEARCH_DATA_DIR:/usr/share/opensearch/data" \
  --entrypoint bash \
  "$OPENSEARCH_IMAGE" \
  -lc "mkdir -p /usr/share/opensearch/data && chown -R 1000:1000 /usr/share/opensearch/data && chmod -R ug+rwX /usr/share/opensearch/data && ls -ld /usr/share/opensearch/data"

sudo docker run --rm \
  --user 1000:1000 \
  -v "$OPENSEARCH_DATA_DIR:/usr/share/opensearch/data" \
  --entrypoint bash \
  "$OPENSEARCH_IMAGE" \
  -lc "id && mkdir -p /usr/share/opensearch/data/nodes && touch /usr/share/opensearch/data/write-test && ls -la /usr/share/opensearch/data"

sudo docker run -d \
  --name "$OPENSEARCH_CONTAINER" \
  --user 1000:1000 \
  -p 9200:9200 \
  -p 9600:9600 \
  --memory="16gb" \
  --memory-swap="16gb" \
  --cpus="16" \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=65536:65536 \
  --cap-add=IPC_LOCK \
  -v "$OPENSEARCH_DATA_DIR:/usr/share/opensearch/data" \
  -e "discovery.type=single-node" \
  -e "bootstrap.memory_lock=true" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  -e "OPENSEARCH_JAVA_OPTS=-Xms8g -Xmx8g" \
  "$OPENSEARCH_IMAGE"

echo "OpenSearch started."
echo "Data dir: $OPENSEARCH_DATA_DIR"