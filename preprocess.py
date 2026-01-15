# preprocess.py
"""
STEP 2: DATA PREPROCESSING & CHUNKING LAYER
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import List


# ===============================
# Configuration
# ===============================

OUTPUT_DIR = "scraped_data"
DEFAULT_CHUNK_SIZE = 250   
DEFAULT_OVERLAP = 40


# ===============================
# Utility Functions
# ===============================

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(words: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap

    return chunks


# ===============================
# Core Processing Logic
# ===============================

def preprocess_scraped_data(
    data: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> dict:

    pages = data.get("site_pages", {}).get("pages", [])

    all_chunks = []
    chunk_metadata = []

    for page in pages:
        page_url = page.get("page_url")

        title = page.get("title", "")
        headings = " ".join(h["text"] for h in page.get("headings", []))
        body = page.get("full_text", "")

        # 🔥 ALWAYS combine all signals
        combined_text = clean_text(
            " ".join([title, headings, body])
        )

        # Skip only truly useless pages
        if len(combined_text.split()) < 40:
            continue

        words = combined_text.split()
        chunks = chunk_text(words, chunk_size, overlap)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "page_url": page_url,
                "chunk_index": idx
            })

    return {
        "source_url": data.get("source_url"),
        "processed_at": datetime.utcnow().isoformat() + "Z",
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
        "chunk_metadata": chunk_metadata
    }


# ===============================
# File IO
# ===============================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processed_output(processed: dict, input_path: str) -> str:
    base = os.path.basename(input_path).replace(".json", "")
    output_name = f"{base}_processed.json"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    return output_path


# ===============================
# Main
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Preprocess scraped website data for RAG")
    parser.add_argument("--input", required=True, help="Path to scraped JSON file")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"[INFO] Loading scraped data: {args.input}")
    raw_data = load_json(args.input)

    print("[INFO] Preprocessing & chunking text...")
    processed = preprocess_scraped_data(
        raw_data,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

    output_path = save_processed_output(processed, args.input)

    print(f"[SUCCESS] Processed data saved to {output_path}")
    print(f"[INFO] Total chunks created: {processed['total_chunks']}")


if __name__ == "__main__":
    main()
