"""Validate that the sample corpus can be ingested and indexed."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.service import RagService


if __name__ == "__main__":
    service = RagService(ROOT / "data" / "docs")
    count = service.ingest()
    print(f"Indexed chunks: {count}")
