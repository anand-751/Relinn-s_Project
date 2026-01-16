import argparse
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()


# ===============================
# Configuration
# ===============================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8


# ===============================
# Core Logic
# ===============================

def load_vectorstore(index_path: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def build_prompt():
    return ChatPromptTemplate.from_template(
        """
You are a helpful AI assistant.
Use the context below to answer the question.

- If the context contains partial information, infer a helpful answer.
- Do NOT copy headings verbatim.
- Explain in clear sentences.
- dont give the content's references
- If the answer is not present at all, say:
  "I don't know about this

Context:
{context}

Question:
{question}

Answer (in your own words):
"""
    )



def run_chatbot(vectorstore: FAISS):
    
    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    prompt = build_prompt()
    parser = StrOutputParser()

    print("\n🤖 RAG Chatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        docs = vectorstore.similarity_search(question, k=TOP_K)
        context = "\n\n".join(d.page_content for d in docs)

        chain = prompt | llm | parser
        answer = chain.invoke({"context": context, "question": question})

        print(f"\nBot: {answer}\n")


# ===============================
# Main Execution
# ===============================

def main():
    parser = argparse.ArgumentParser(description="Console RAG chatbot using FAISS + Groq")
    parser.add_argument("--index", required=True, help="Path to FAISS index directory")
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Index path not found: {args.index}")

    print(f"[INFO] Loading FAISS index from {args.index}")
    vectorstore = load_vectorstore(args.index)

    run_chatbot(vectorstore)


if __name__ == "__main__":
    main()
