from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from scripts.embedder import ST_Embedder
from scripts.searcher import HybridSearcher
from scripts.filters import llm_to_filters

load_dotenv()

app = FastAPI(title="Araç Satış Asistanı API")

# ======================
# Request & Response
# ======================
class QueryRequest(BaseModel):
    query: str
    history: List[str] = Field(default_factory=list)

class CarResult(BaseModel):
    yil: Optional[int]
    marka: Optional[str]
    seri: Optional[str]
    model: Optional[str]
    fiyat: Optional[float]
    kilometre: Optional[float]
    yakit_tipi: Optional[str]
    vites_tipi: Optional[str]
    url: Optional[str]
    description: str


# ======================
# Init
# ======================
client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
embedder = ST_Embedder()
searcher = HybridSearcher(client, "car_listings_st", embedder)

TOPIC_THRESHOLD = 0.5
HISTORY_TAKE = 3


# ======================
# Helpers
# ======================
def detect_strict_mode(query: str) -> bool:
    """Benzer / alternatif gibi ifadelerde strict=False yap."""
    keywords = ["benzer", "alternatif", "farklı"]
    return not any(k in query.lower() for k in keywords)


def is_new_topic(query: str, history: List[str], threshold: float = TOPIC_THRESHOLD) -> bool:
    """Mesaj geçmişine göre yeni konu mu kontrol et."""
    if not history:
        return False

    q_emb = np.array(embedder.encode([query])[0]).reshape(1, -1)
    h_embs = np.array(embedder.encode(history[-HISTORY_TAKE:]))

    sims = cosine_similarity(q_emb, h_embs)
    return float(np.max(sims)) < threshold


def in_tolerance(value: Optional[float], target: float, tol: float = 0.05) -> bool:
    """Değer, hedefin ±%tol aralığında mı?"""
    if value is None:
        return False
    return (target * (1 - tol)) <= value <= (target * (1 + tol))


# ======================
# Endpoint
# ======================
@app.post("/search", response_model=List[CarResult])
def search(req: QueryRequest):
    # 1) Konu algılama
    history = req.history or []
    if is_new_topic(req.query, history):
        history = []

    # 2) LLM filtreleri
    contextual_query = f"Kullanıcı geçmişi: {history}. Yeni mesaj: {req.query}"
    filters = llm_to_filters(contextual_query)

    # 3) Strict mode
    strict = detect_strict_mode(req.query)

    # 4) Qdrant araması (yüksek top_k → daha fazla aday araç)
    search_text = req.query
    results = searcher.search(search_text, f=filters, top_k=100, strict=strict)

    if not results:
        print("⚠️ Strict aramada sonuç çıkmadı, fallback strict=False")
        results = searcher.search(search_text, f=filters, top_k=100, strict=False)

    # 5) Kullanıcının hedef fiyat ve km
    target_price = None
    target_km = None

    if filters.fiyat_max and not filters.fiyat_min:
        target_price = filters.fiyat_max
    elif filters.fiyat_min and not filters.fiyat_max:
        target_price = filters.fiyat_min
    elif filters.fiyat_min and filters.fiyat_max:
        target_price = (filters.fiyat_min + filters.fiyat_max) / 2

    if filters.km_max and not filters.km_min:
        target_km = filters.km_max
    elif filters.km_min and not filters.km_max:
        target_km = filters.km_min
    elif filters.km_min and filters.km_max:
        target_km = (filters.km_min + filters.km_max) / 2

    # 6) Sonuçları dönüştür
    cars: List[CarResult] = []
    for _, _, pl in results:
        fiyat = pl.get("fiyat")
        km = pl.get("kilometre")
        yil = pl.get("yil")
        marka = pl.get("marka")
        model = pl.get("model")
        seri = pl.get("seri")

        fiyat_str = f"{fiyat:,}".replace(",", ".") if fiyat else "bilinmiyor"
        km_str = f"{int(km):,}".replace(",", ".") if km else "bilinmiyor"

        desc = (
            f"**{yil or '—'} model {marka or '—'} {model or ''} {seri or ''}**\n"
            f"- Fiyat: {fiyat_str} TL\n"
            f"- Kilometre: {km_str} km\n"
            f"- Yakıt: {pl.get('yakit_tipi', 'bilinmiyor')}\n"
            f"- Vites: {pl.get('vites_tipi', 'bilinmiyor')}\n"
            f"- 👉 [İlana Git]({pl.get('url')})"
        )

        cars.append(CarResult(
            yil=yil,
            marka=marka,
            seri=seri,
            model=model,
            fiyat=fiyat,
            kilometre=km,
            yakit_tipi=pl.get("yakit_tipi"),
            vites_tipi=pl.get("vites_tipi"),
            url=pl.get("url"),
            description=desc
        ))

    # 7) Eğer sort_by varsa → direk onu uygula
    if hasattr(filters, "sort_by") and filters.sort_by:
        if filters.sort_by == "fiyat_desc":
            cars = sorted(cars, key=lambda c: c.fiyat or 0, reverse=True)
        elif filters.sort_by == "fiyat_asc":
            cars = sorted(cars, key=lambda c: c.fiyat or 0)
        elif filters.sort_by == "yil_desc":
            cars = sorted(cars, key=lambda c: c.yil or 0, reverse=True)
        elif filters.sort_by == "yil_asc":
            cars = sorted(cars, key=lambda c: c.yil or 0)
        elif filters.sort_by == "km_desc":
            cars = sorted(cars, key=lambda c: c.kilometre or 0, reverse=True)
        elif filters.sort_by == "km_asc":
            cars = sorted(cars, key=lambda c: c.kilometre or 0)
    else:
        # 8) sort_by yoksa → toleranslı sıralama uygula
        if target_price:
            cars = sorted(
                cars,
                key=lambda c: (
                    0 if in_tolerance(c.fiyat, target_price, tol=0.05) else 1,
                    abs((c.fiyat or 0) - target_price) if c.fiyat else float("inf")
                )
            )
        if target_km:
            cars = sorted(
                cars,
                key=lambda c: (
                    0 if in_tolerance(c.kilometre, target_km, tol=0.05) else 1,
                    abs((c.kilometre or 0) - target_km) if c.kilometre else float("inf")
                )
            )

    # 🔑 sadece en uygun 5 aracı döndür
    return cars[:5]
