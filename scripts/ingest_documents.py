"""
CLI script: bulk ingest a folder or file into the Chroma vector store.

Usage:
    python scripts/ingest_documents.py --dir ./my_docs
    python scripts/ingest_documents.py --file ./report.pdf
    python scripts/ingest_documents.py --dir ./docs --collection my_collection
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aetheris.rag.ingest import SUPPORTED_EXTENSIONS, ingest_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into AETHERIS vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="Directory containing documents to ingest")
    group.add_argument("--file", type=Path, help="Single file to ingest")
    parser.add_argument("--collection", default="aetheris", help="Chroma collection name")
    args = parser.parse_args()

    files: list[Path] = []
    if args.file:
        files = [args.file]
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(args.dir.rglob(f"*{ext}"))

    if not files:
        print("No supported documents found.")
        sys.exit(0)

    print(f"Found {len(files)} document(s) to ingest into collection '{args.collection}'")

    for path in files:
        try:
            result = ingest_file(path, collection_name=args.collection)
            print(f"  OK  {path.name} → {result.n_chunks} chunks (id={result.document_id[:8]}…)")
        except Exception as exc:
            print(f"  ERR {path.name}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
