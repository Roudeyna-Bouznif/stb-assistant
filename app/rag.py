import os
import httpx
import chromadb
from app import config

client = chromadb.PersistentClient(path=config.CHROMA_DIR)
collection = client.get_or_create_collection("stb_kb")


def embed(texts: list[str], task: str) -> list[list[float]]:
    """Embeddings Gemini par lots de 50."""
    out = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        body = {"requests": [{
            "model": f"models/{config.EMBED_MODEL}",
            "content": {"parts": [{"text": t}]},
            "taskType": task,
        } for t in batch]}
        r = httpx.post(
            f"{config.BASE_URL}/models/{config.EMBED_MODEL}:batchEmbedContents",
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json=body, timeout=60,
        )
        r.raise_for_status()
        out += [e["values"] for e in r.json()["embeddings"]]
    return out


def chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Découpe par paragraphes, regroupés jusqu'à ~size caractères."""
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) > size and current:
            chunks.append(current.strip())
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:]   # repartir au début d'un mot
            current = tail + "\n\n" + para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def index_documents():
    ids, docs, metas = [], [], []
    for name in sorted(os.listdir(config.DATA_DIR)):
        if not name.endswith((".md", ".txt")):
            continue
        with open(os.path.join(config.DATA_DIR, name), encoding="utf-8") as f:
            for n, c in enumerate(chunk(f.read())):
                ids.append(f"{name}-{n}")
                docs.append(c)
                metas.append({"source": name})
    if not docs:
        print("Aucun document trouvé dans data/")
        return
    collection.upsert(ids=ids, documents=docs, metadatas=metas,
                      embeddings=embed(docs, "RETRIEVAL_DOCUMENT"))
    print(f"{len(docs)} chunks indexés.")


def search(question: str, k: int = 4):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    res = collection.query(query_embeddings=[q], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


if __name__ == "__main__":
    index_documents()
    for doc, meta, dist in search("Quels documents pour ouvrir un compte ?"):
        print(f"[{meta['source']}] ({dist:.3f}) {doc[:120]}...")