"""
filters.py
LLM tabanlı filtre çıkarıcı
- Kullanıcının doğal dildeki sorgusunu alır
- JSON formatında filtre çıkarır (marka, fiyat, yıl, km, yakıt, vites, vb.)
- QueryFilters objesine dönüştürür
"""

import os
from typing import Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from scripts.qdrant_utils import QueryFilters


# ============================
# LLM çıktısı için geçici model
# ============================
class FilterSpec(BaseModel):
    """
    LLM’den gelen JSON’un parse edileceği model.
    Daha sonra QueryFilters içine aktarılır.
    """
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
# Prompt (LLM talimatı)
# ============================
SYSTEM = """
Sen bir araç arama filtresi çıkarıcısısın.
Görev: Kullanıcının doğal dil sorgusundan yapısal filtre JSON'u çıkar.

Kurallar:
- SADECE JSON döndür.
- Açıklama yazma, sadece JSON.
- Alanlar: marka, seri, model, konum,
  fiyat_min, fiyat_max, yil_min, yil_max,
  km_min, km_max, yakit, vites, sort_by.
- Fiyat için:
  "en fazla, altı, kadar" → fiyat_max
  "en az, üstü, fazla" → fiyat_min
- Yıl için:
  "sonrası, üstü, yeni" → yil_min
  "öncesi, altı, eski" → yil_max
- Kilometre için:
  "altı, en fazla, düşük" → km_max
  "üstü, fazla, yüksek" → km_min
- Eğer yeni mesaj sadece bütçe, yıl, km, yakıt veya vites kriteri içeriyorsa,
  önceki marka/seri/model seçimini koru.
- Yakıt/vites kullanıcı isterse ekle, yoksa boş bırak.
- Yazım farklılıkları olabilir ("benzin" vs "benzinli"), normalize etme, gelen ifadeyi aynen al.
- "en pahalı", "en yüksek fiyatlı" → sort_by = "fiyat_desc"
- "en ucuz", "en düşük fiyatlı" → sort_by = "fiyat_asc"
- "en yeni", "en güncel" → sort_by = "yil_desc"
- "en eski" → sort_by = "yil_asc"
- "en az km", "düşük km" → sort_by = "km_asc"
- "en çok km", "yüksek km" → sort_by = "km_desc"
"""

HUMAN = """
Sorgu: "{query}".
Lütfen sadece JSON döndür (şema: {format_instructions}).
"""

# Prompt zinciri
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM.strip()),
    ("human", HUMAN.strip()),
])

# Parser
parser = PydanticOutputParser(pydantic_object=FilterSpec)


# ============================
# Ana fonksiyon
# ============================
def llm_to_filters(query: str, model: str = "gpt-4o-mini") -> QueryFilters:
    """
    Kullanıcı sorgusunu LLM'e gönderir, JSON filtre çıkarır ve QueryFilters döner.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmadı!")

    # LLM
    llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)

    # Zincir (prompt → LLM → parser)
    chain = prompt | llm | parser

    # Çalıştır
    spec: FilterSpec = chain.invoke({
        "query": query,
        "format_instructions": parser.get_format_instructions()
    })

    # FilterSpec → QueryFilters dönüşümü
    return QueryFilters(**spec.dict())
