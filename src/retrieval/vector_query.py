import chromadb
from sentence_transformers import SentenceTransformer


def semantic_query(query_text: str, top_k: int = 10) -> list[dict]:
    client = chromadb.PersistentClient(path="../../data/vectors")
    model = SentenceTransformer('BAAI/bge-m3')
    collection = client.get_collection("embeddings")

    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return results