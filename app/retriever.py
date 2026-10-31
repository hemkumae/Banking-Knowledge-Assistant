import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "banking_docs.json"


class Retriever:
    def __init__(self):
        self.documents = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        texts = [doc["text"] for doc in self.documents]
        vectors = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors.astype("float32"))

    def search(self, query: str, k: int = 3):
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.index.search(
            query_vector.astype("float32"), min(k, len(self.documents))
        )
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                document = dict(self.documents[int(index)])
                document["score"] = float(score)
                results.append(document)
        return results
