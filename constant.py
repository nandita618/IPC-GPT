import os
from chromadb.config import Settings



chroma_settings = Settings(
    persist_directory="db",
    anonymized_telemetry=False
)
