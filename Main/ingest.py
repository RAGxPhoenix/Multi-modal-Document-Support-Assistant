
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_index(data_dir="data/kb_docs", storage_dir="storage"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SystemExit(f"❌ Data directory {data_dir} does not exist. Create it and put PDFs there.")

    print(f"📂 Loading documents from: {data_dir}")
    documents = SimpleDirectoryReader(str(data_dir)).load_data()
    print(f"✅ Loaded {len(documents)} documents (may be split into multiple chunks).")

    # HuggingFace embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model  # set global embeddings

    # Build vector index
    print("⚙️ Building vector index...")
    index = VectorStoreIndex.from_documents(documents)

    # Persist the index
    print("💾 Saving index to disk...")
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"✅ Index saved into {storage_dir}/")


if __name__ == "__main__":
    build_index()
