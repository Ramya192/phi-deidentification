#!/usr/bin/env bash
# Entrypoint for the Hugging Face Space container. Starts the FastAPI
# backend on localhost:8000 in the background, waits for it to answer
# /health, then runs Streamlit in the foreground on the container's public
# port (7860) -- Streamlit's default sidebar API base URL already points
# at http://localhost:8000, so no config is needed for the UI to find it.
set -e

if [ -z "${PHI_DEID_API_KEY}" ]; then
  echo "WARNING: PHI_DEID_API_KEY is not set as a Space secret."
  echo "The API will auto-generate a random key for this session, and the"
  echo "Streamlit sidebar will not be able to pre-fill it, which means every"
  echo "request will 401. Set it under Settings -> Variables and secrets."
fi

uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "Waiting for the API to come up..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "API is up."
    break
  fi
  sleep 1
done

streamlit run app/streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false

# If Streamlit exits, bring the API down too so the container actually stops.
kill "$API_PID" 2>/dev/null || true
