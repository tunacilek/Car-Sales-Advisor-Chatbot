"""
qdrant_utils.py
Qdrant yardımcı fonksiyonları
- ID üretme
- Arama metni oluşturma
- Payload hazırlama
- Koleksiyon oluşturma
- DataFrame → Qdrant upsert etme
- QueryFilters modeli (LLM çıktısını tutmak için)
- QueryFilters → Qdrant Filter dönüşümü (sadece sayısal alanlar: fiyat, yıl, km)
"""

import uuid
from typing import Any, Dict, Optional, List
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
)
from pydantic import BaseModel

from scripts.normalize import ascii_lower, extract_city


# ============================
# ID üretici
# ============================
def make_point_id(raw: Any) -> Any:
    try:
        return int(raw)
    except Exception:
        return str(uuid.uuid4())


# ============================
# Doküman metni
# ============================
def build_doc_text(r: Dict[str, Any]) -> str:
    return (
        f"{r.get('marka')} {r.get('seri')} {r.get('model')} {r.get('yil')} – "
        f"{r.get('yakit_tipi')}, {r.get('vites_tipi')}, {r.get('kilometre')} km, "
        f"{r.get('kasa_tipi')}, {r.get('konum')}. "
        f"Fiyat: {r.get('fiyat')}."
    )


# ============================
# Payload hazırlayıcı
# ============================
def build_payload(r: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Qdrant payload objesi üretir.
    Konum için şehir çıkarılır, diğer alanlar ascii_lower yapılır.
    """
    return {
        **r,
        "marka_key": ascii_lower(r.get("marka")),
        "seri_key": ascii_lower(r.get("seri")),
        "model_key": ascii_lower(r.get("model")),
        "konum_key": extract_city(r.get("konum")),
        "yakit_key": ascii_lower(r.get("yakit_tipi")),
        "vites_key": ascii_lower(r.get("vites_tipi")),
        "text": text,
    }


# ============================
# Koleksiyon kontrol/oluştur
# ============================
def ensure_collection(client: QdrantClient, name: str, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


# ============================
# DataFrame → Qdrant Upsert
# ============================
def df_to_points(
    df: pd.DataFrame,
    embedder,
    client: QdrantClient,
    collection: str,
    batch_size: int = 256,
):
    ensure_collection(client, collection, embedder.dimension())
    rows = df.to_dict(orient="records")

    for i in tqdm(range(0, len(rows), batch_size), desc="Upserting to Qdrant"):
        chunk = rows[i : i + batch_size]
        texts = [build_doc_text(r) for r in chunk]
        vecs = embedder.embed_documents(texts)

        points = [
            PointStruct(
                id=make_point_id(r["id"]),
                vector=v,
                payload=build_payload(r, t),
            )
            for r, v, t in zip(chunk, vecs, texts)
        ]

        client.upsert(collection_name=collection, points=points)


# ============================
# QueryFilters (LLM çıkışı için)
# ============================
class QueryFilters(BaseModel):
    marka: Optional[str] = None
    seri: Optional[str] = None
    model: Optional[str] = None
    konum: Optional[str] = None
    fiyat_min: Optional[float] = None
    fiyat_max: Optional[float] = None
    yil_min: Optional[int] = None
    yil_max: Optional[int] = None
    km_min: Optional[float] = None   # ✅ kilometre alt sınır
    km_max: Optional[float] = None   # ✅ kilometre üst sınır
    yakit: Optional[str] = None
    vites: Optional[str] = None
    sort_by: Optional[str] = None


# ============================
# QueryFilters → Qdrant Filter
# ============================
def build_qdrant_filter(f: QueryFilters) -> Optional[Filter]:
    """
    Sadece sayısal alanları filtreler:
    - fiyat_min / fiyat_max
    - yil_min / yil_max
    - km_min / km_max

    Marka, seri, model, konum, yakit, vites → semantic search üzerinden yakalanır.
    """
    must: List[FieldCondition] = []

    def rng(field: str, gte=None, lte=None):
        cond = {}
        if gte is not None:
            cond["gte"] = float(gte)
        if lte is not None:
            cond["lte"] = float(lte)
        if cond:
            must.append(FieldCondition(key=field, range=Range(**cond)))

    # ✅ Sayısal filtreler
    rng("fiyat_num", gte=f.fiyat_min, lte=f.fiyat_max)
    rng("yil_num", gte=f.yil_min, lte=f.yil_max)
    rng("km_num", gte=f.km_min, lte=f.km_max)

    return Filter(must=must) if must else None


client = QdrantClient("http://localhost:6333")

# Koleksiyon adını kendi koleksiyonuna göre değiştir
collection_name = "car_listings_st"

