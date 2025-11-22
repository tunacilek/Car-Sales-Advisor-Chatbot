"""
normalize.py
Veri ön işleme ve normalize etme fonksiyonları
- Türkçe karakterleri ASCII yapma
- Küçük harfe çevirme
- Fiyat / kilometre -> sayıya çevirme
- Yıl bilgisini çıkarma
- Konumdan şehir bilgisini alma
- DataFrame normalize etme
"""

import re
import pandas as pd
from unidecode import unidecode


# ============================
# Küçük harfe çevir + Türkçe karakterleri ASCII yap
# ============================
def ascii_lower(s: str) -> str:
    return unidecode(str(s or "")).strip().lower()


# ============================
# Fiyat ve kilometre için sayıya çevir
# ============================
def to_num(text):
    if text is None:
        return None
    s = str(text).lower()
    s = s.replace("tl", "").replace("₺", "").replace("km", "")
    s = s.replace(".", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


# ============================
# Yıl bilgisini 4 haneli yakala
# ============================
def year4(x):
    m = re.search(r"\b(19|20)\d{2}\b", str(x or ""))
    return int(m.group(0)) if m else None


# ============================
# Konumdan şehir çıkar
# ============================
def extract_city(konum: str) -> str:
    """
    Konum string'inden şehir bilgisini döndürür.
    Örn: "Karşıyaka Mh. Kepez, Antalya" -> "antalya"
    """
    if not konum:
        return ""
    return str(konum).split(",")[-1].strip().lower()


# ============================
# Ana normalize fonksiyonu
# ============================
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame içindeki kolonları normalize eder.
    - fiyat, kilometre, yıl → sayısal alan
    - marka, seri, model, konum, kasa_tipi, cekis → key alanı
    """
    df = df.copy()

    # Sayısal dönüşümler
    df["fiyat_num"] = df["fiyat"].map(to_num)
    df["km_num"] = df["kilometre"].map(to_num)
    df["yil_num"] = df["yil"].map(year4)

    # Eşleşme için küçük harfli key alanları
    for col in ["marka", "seri", "model", "konum", "kasa_tipi", "cekis"]:
        df[col + "_key"] = df[col].map(ascii_lower)

    # Konum özel → şehir bilgisi
    df["konum_key"] = df["konum"].map(extract_city)

    return df
