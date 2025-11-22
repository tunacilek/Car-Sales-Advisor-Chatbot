import os
from typing import List, Dict
from langchain_openai import ChatOpenAI

def format_car_results_stream(user_query: str, cars: List[Dict]):
    """
    Araç listesini LLM üzerinden satış danışmanı tarzında stream ederek formatlar.
    Her aracı ayrı blok halinde sunar, sonunda kısa bir kıyaslama + öneri yapar.
    Yields: parça parça string (streaming için).
    """
    if not cars:
        yield "Sana uygun araç bulamadım. 😕 Başka bir şey sorabilirsin."
        return

    # Araçları LLM’e gidecek string haline getir
    cars_text = "\n\n".join([
        f"- {car.get('yil', '—')} model {car.get('marka', '—')} {car.get('seri','')} {car.get('model','')} | "
        f"Fiyat: {car.get('fiyat','bilinmiyor')} TL | "
        f"Kilometre: {car.get('kilometre','bilinmiyor')} km | "
        f"Yakıt: {car.get('yakit_tipi','bilinmiyor')} | "
        f"Vites: {car.get('vites_tipi','bilinmiyor')} | "
        f"URL: {car.get('url','')}"
        for car in cars
    ])

    # --- Daha sade ve doğru sistem prompt ---
    system_prompt = """
Sen bir araç satış danışmanısın.
Kullanıcının sorgusuna uygun araçları düzenli, kolay okunabilir bir şekilde listele.
Her aracı ayrı bir blok halinde sun.

Format:
### {YIL} {MARKA} {MODEL}
- Fiyat: ...
- Kilometre: ...
- Yakıt: ...
- Vites: ...
- 👉 İlana Git

Sonunda:
- Araçlar arasında kısa bir kıyaslama yap (maksimum 3 cümle).
- Tavsiyeni daima 'Ben senin yerinde olsam...' şeklinde ver.
- 'Eğer benim yerimde olsan...' ifadesini KULLANMA.
- Avantaj / Dezavantaj listeleri YAZMA.
"""

    human_prompt = f"Kullanıcının sorgusu: {user_query}\n\nAday araçlar:\n{cars_text}"

    # OpenAI LLM
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",       # güçlü model
        temperature=0.3,
        streaming=True
    )

    # streaming → parça parça yield et
    for chunk in llm.stream([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt}
    ]):
        if chunk.content:
            yield chunk.content
