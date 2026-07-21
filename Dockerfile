# Single-container deployment for Hugging Face Spaces (Docker SDK).
# Runs both the FastAPI backend and the Streamlit UI in one container --
# the UI talks to the API over localhost, exactly the same HTTP-client
# architecture described in README.md ("Run the Streamlit UI"), just with
# both processes co-located so a free-tier Space only needs to expose one
# public port (7860, Streamlit's).
#
# Uses en_core_web_lg (not the smaller _sm model) deliberately: it's the
# model requirements.txt documents as primary and the one every eval
# number in the final report was produced with. Swapping to en_core_web_sm
# here would make the live demo behave differently from what's written up
# -- if a build timeout ever forces that trade-off, see the commented
# fallback line below and update README's eval section to match.
FROM python:3.11-slim

WORKDIR /app

# Minimal build tooling for any package that needs to compile from source,
# plus curl for start.sh's API-readiness healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Primary model -- matches what the eval numbers in the report were run with.
RUN python -m spacy download en_core_web_lg
# Fallback if a build ever needs to be faster/lighter (uncomment instead of
# the line above, and update Presidio's NlpEngineProvider config plus the
# README eval section to note the swap):
# RUN python -m spacy download en_core_web_sm

COPY . .

# PHI_DEID_API_KEY is intentionally NOT set here -- set it as a "Secret" in
# the Space's Settings instead (Settings -> Variables and secrets -> New
# secret). Leaving it truly unset in the image lets api/main.py's existing
# auto-generate-and-warn fallback catch the case where someone forgets to
# configure it, rather than silently running with an empty-string key.

# Only Streamlit's port needs to be public; the API stays on localhost:8000
# inside the container and is reached there by the UI's default API base URL.
EXPOSE 7860

RUN chmod +x start.sh
CMD ["./start.sh"]
