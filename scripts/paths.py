import os

# Get the root directory - handle both local development and Streamlit Cloud
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# If we're on Streamlit Cloud, use a more appropriate directory
if os.path.exists("/mount/src"):
    # Streamlit Cloud environment
    ROOT_DIR = "/mount/src/ragstream"
    OUTPUTS_DIR = "/tmp/ragstream_outputs"  # Use /tmp for writable directory
else:
    # Local development environment
    OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

ENV_FPATH = os.path.join(ROOT_DIR, ".env")

CODE_DIR = os.path.join(ROOT_DIR, "scripts")

APP_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "config.yaml")
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")

DATA_DIR = os.path.join(ROOT_DIR, "data")
PUBLICATION_FPATH = os.path.join(DATA_DIR, "publication.md")

VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")

CHAT_HISTORY_DB_FPATH = os.path.join(OUTPUTS_DIR, "chat_history.db")