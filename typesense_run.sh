sudo docker run -p 8108:8108 --memory="16gb" --memory-swap="16gb" --cpus="16" -v/tmp:/data typesense/typesense:30.2 --data-dir /data --api-key=xyz
