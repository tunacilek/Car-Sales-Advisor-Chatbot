# scripts/deneme.py
import pandas as pd
from qdrant_client import QdrantClient
from scripts.normalize import normalize_df
from scripts.embedder import ST_Embedder
from scripts.qdrant_utils import df_to_points

if __name__ == "__main__":
    # ===============================
    # 1) Veriyi oku
    # ===============================
    df = pd.read_parquet("data/arabam_ilanlar.parquet")
    print(f"Orijinal veri şekli: {df.shape}")

    # ===============================
    # 2) Normalize et
    # ===============================
    df = normalize_df(df)
    print(f"Normalize edilmiş kolonlar: {df.columns.tolist()}")

    # ===============================
    # 3) Qdrant Client
    # ===============================
    client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

    # ===============================
    # 4) Embedder
    # ===============================
    embedder = ST_Embedder()

    # ===============================
    # 5) Qdrant’a yükle (test için ilk 100 satır)
    # ===============================
    # scripts/deneme.py
    df_to_points(df, embedder, client, "car_listings_st", batch_size=256)
    print("✅ Tüm kayıtlar Qdrant’a yüklendi.")

