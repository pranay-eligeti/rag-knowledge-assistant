"""Validate that the sample corpus can be ingested and indexed."""

from pathlib import Path

from src.service import RagService


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    service = RagService(root / "data" / "docs")
    count = service.ingest()
    print(f"Indexed chunks: {count}")
