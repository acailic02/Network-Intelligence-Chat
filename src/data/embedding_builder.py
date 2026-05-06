import chromadb
from sentence_transformers import SentenceTransformer
import json
from huggingface_hub import login
from src.config import HF_TOKEN

login(token=HF_TOKEN)

client = chromadb.PersistentClient(path="../../data/vectors")
model = SentenceTransformer('BAAI/bge-m3')

def build_embeddings():
    with open("../../data/snapshot.json", 'r') as f:
        snapshot = json.load(f)
        text = []
        ids = []
        metadatas = []
        for connection in snapshot:
            person = connection["enrichment"].get("person", {})
            summary = person.get("summary", "") or ""
            headline = person.get("headline", "") or ""

            text.append("Headline: " + headline + '\n' + "Summary: " + summary)
            ids.append(connection["source_row"]["url"])
            metadatas.append({
                "first_name": connection["source_row"]["first_name"],
                "last_name": connection["source_row"]["last_name"],
                "headline": headline,
                "summary": summary,
                "linkedin_url": connection["source_row"]["url"],
                "owners": ", ".join(connection["owners"])
            })

    embeddings = model.encode(text)

    collection = client.create_collection("embeddings")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=text,
        ids=ids,
        metadatas=metadatas,
    )

if __name__ == "__main__":
    build_embeddings()




