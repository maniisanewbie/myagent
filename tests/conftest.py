import os

# main.py builds the provider at import time, which requires credentials to exist.
os.environ.setdefault("MYAGENT_API_KEY", "test-key")
