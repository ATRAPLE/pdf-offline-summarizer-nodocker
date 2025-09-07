import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

CHROMA_PERSIST = os.getenv("CHROMA_PERSIST", "./.chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "pdf_chunks")

EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL_NAME)

class VectorIndex:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST, settings=Settings(allow_reset=True))
        self.col = self.client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=_ef)

    def reset(self):
        self.client.reset()
        self.col = self.client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=_ef)

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        ids = [f"doc-{m['job_id']}-{m['seq']}" for m in metadatas]
        self.col.add(documents=chunks, metadatas=metadatas, ids=ids)

    def query(self, query: str, k: int = 5):
        return self.col.query(query_texts=[query], n_results=k)
