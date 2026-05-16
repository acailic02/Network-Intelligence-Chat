import chromadb
from sentence_transformers import SentenceTransformer
import os

def semantic_query(query_text: str, filters: dict, top_k: int = 10) -> list[dict]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "..", "..", "data", "vectors")

    client = chromadb.PersistentClient(path=db_path)

    model = SentenceTransformer('BAAI/bge-m3')
    collection = client.get_collection("embeddings")

    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, where=filters, n_results=top_k)

    return results