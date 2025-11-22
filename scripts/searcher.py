"""
HybridSearcher (semantic ağırlıklı)
- Kullanıcı sorgusunu embedding'e dönüştürür
- Qdrant'ta arama yapar (dense + filtreler)
"""

from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint, Filter, FieldCondition, MatchValue

from scripts.qdrant_utils import QueryFilters, build_qdrant_filter


class HybridSearcher:
    def __init__(self, client: QdrantClient, collection: str, embedder):
        """
        - client: QdrantClient örneği
        - collection: Qdrant koleksiyon adı
        - embedder: SentenceTransformer benzeri bir model
        """
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def search(
        self,
        query: str,
        f: Optional[QueryFilters] = None,
        top_k: int = 10,
        strict: bool = False
    ) -> List[Tuple[str, float, dict]]:
        """
        Arama yap:
        - query → embedding
        - Qdrant search
        - Sadece sayısal filtreler (fiyat / yıl / km)
        - Eğer strict=True ise: marka/seri/model de filtrelenir
        """
        # 1) Query → embedding
        query_vec = self.embedder.embed_query(query)

        # 2) Sayısal filtreleri hazırla
        qdrant_filter = build_qdrant_filter(f) if f else None

        # 3) Eğer strict=True → marka/seri/model de ekle
        if f and strict:
            must = qdrant_filter.must if qdrant_filter else []

            def eq(field: str, val: Optional[str]):
                if val:
                    must.append(FieldCondition(key=field, match=MatchValue(value=val.lower().strip())))

            eq("marka_key", f.marka)
            eq("seri_key", f.seri)
            eq("model_key", f.model)

            qdrant_filter = Filter(must=must) if must else None

        if f:
            print("📌 Uygulanan filtreler:", f.model_dump())
        if qdrant_filter:
            print("✅ Qdrant filtresi aktif")

        # 4) Qdrant search
        res: List[ScoredPoint] = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            query_filter=qdrant_filter,
            limit=top_k,
        )

        # 5) (id, skor, payload) döndür
        return [(str(p.id), p.score, p.payload) for p in res]
