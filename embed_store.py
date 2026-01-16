import argparse
import json
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ===============================
# Configuration
# ===============================

VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ===============================
# Utility Functions
# ===============================

def load_processed_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(chunks: List[str], metadata: List[dict]) -> List[Document]:
    documents = []
    for text, meta in zip(chunks, metadata):
        documents.append(
            Document(
                page_content=text,
                metadata=meta
            )
        )
    return documents


# ===============================
# Core Logic
# ===============================

def create_faiss_index(processed_data: dict) -> FAISS:
    print("[INFO] Initializing embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )


    print("[INFO] Building LangChain documents...")
    documents = build_documents(
        processed_data["chunks"],
        processed_data["chunk_metadata"]
    )

    print(f"[INFO] Creating FAISS index with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


def save_faiss_index(vectorstore: FAISS, index_name: str):
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    path = os.path.join(VECTOR_STORE_DIR, index_name)
    vectorstore.save_local(path)
    print(f"[SUCCESS] FAISS index saved at {path}")


# ===============================
# Main Execution (Console)
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Create FAISS vector store from processed data")
    parser.add_argument("--input", required=True, help="Path to processed JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"[INFO] Loading processed data: {args.input}")
    processed_data = load_processed_data(args.input)

    if processed_data.get("total_chunks", 0) == 0:
        raise ValueError("No chunks found in processed data. Cannot create embeddings.")

    vectorstore = create_faiss_index(processed_data)

    base_name = os.path.basename(args.input).replace("_processed.json", "")
    save_faiss_index(vectorstore, base_name)


if __name__ == "__main__":
    main()
