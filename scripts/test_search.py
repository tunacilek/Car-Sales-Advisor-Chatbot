# scripts/test_search.py
"""
Test Search
- Kullanıcıdan doğal dil sorgusu alır
- LLM → QueryFilters çıkarır
- HybridSearcher ile Qdrant'ta arama yapar
"""

import os
from qdrant_client import QdrantClient

from scripts.embedder import ST_Embedder
from scripts.searcher import HybridSearcher
from scripts.filters import llm_to_filters


if __name__ == "__main__":
    # 1) Qdrant client
    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

    # 2) Embedder (SentenceTransformer)
    embedder = ST_Embedder()

    # 3) Searcher
    searcher = HybridSearcher(client, "car_listings_st", embedder)

    # 4) Kullanıcı sorgusu
    query = "İstanbul’da 1.3 milyon TL’ye kadar 2018 sonrası otomatik benzinli Astra"
    print(f"\n🔎 Kullanıcı Sorgusu: {query}\n")

    # 5) LLM → filtre çıkar
    filters = llm_to_filters(query)
    print("📌 LLM Çıkardığı Filtreler:", filters.model_dump())
    print("📌 LLM Çıkardığı Filtreler (pretty):")
    print(filters, "\n")

    # 6) Arama yap
    results = searcher.search(query, f=filters, top_k=10)

    # 7) Sonuçları yazdır
    print("🔎 Arama Sonuçları:\n")
    if not results:
        print("⚠️ Hiç sonuç bulunamadı.")
    else:
        for pid, score, pl in results:
            print(
                f"ID: {pid} | Score: {score:.4f} | "
                f"Marka: {pl.get('marka')} | Seri: {pl.get('seri')} | "
                f"Model: {pl.get('model')} | Yıl: {pl.get('yil')} | "
                f"Fiyat: {pl.get('fiyat')} | URL: {pl.get('url')}"
            )
