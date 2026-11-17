import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import config
from app.rag import search

app = FastAPI(title="Assistant bancaire (prototype)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # page PHP de démo
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_DISTANCE = 0.75  # au-delà, le passage est jugé non pertinent (à calibrer)
FALLBACK = ("Je n'ai pas trouvé d'information fiable à ce sujet dans ma base "
            "de connaissances. Veuillez contacter votre agence.")

SYSTEM = """Tu es un assistant pour une banque (prototype de démonstration).
Règles strictes :
- Réponds uniquement à partir des extraits fournis. N'utilise aucune connaissance extérieure.
- Si les extraits ne contiennent pas la réponse, dis-le clairement et invite à contacter l'agence.
- N'invente jamais de chiffres, taux, délais ou documents.
- Réponds dans la langue de la question, de façon courte et claire.
- Cite tes sources à la fin sous la forme (source : nom_du_fichier)."""


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question vide")

    try:
        hits = [h for h in search(question) if h[2] <= MAX_DISTANCE]
        if not hits:
            return {"answer": FALLBACK, "sources": []}

        context = "\n\n".join(
            f"[{i + 1}] (source : {meta['source']})\n{doc}"
            for i, (doc, meta, _) in enumerate(hits)
        )
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [
                {"text": f"Extraits :\n{context}\n\nQuestion : {question}"}]}],
            "generationConfig": {"temperature": 0.2},
        }
        r = httpx.post(
            f"{config.BASE_URL}/models/{config.CHAT_MODEL}:generateContent",
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json=body, timeout=60,
        )
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        answer = "".join(p.get("text", "") for p in parts)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Erreur du service IA : {e}")

    return {"answer": answer, "sources": sorted({m["source"] for _, m, _ in hits})}