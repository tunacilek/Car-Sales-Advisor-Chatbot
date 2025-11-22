import streamlit as st
import requests
import sys
import os
import importlib.util

# --- Proje kökü ve formatter.py yolu ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORMATTER_PATH = os.path.join(PROJECT_ROOT, "scripts", "formatter.py")

# --- .env yükle (opsiyonel ama faydalı) ---
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

# --- formatter.py'yi dosya yolundan YÜKLE (pakete bağlı kalmadan) ---
formatter = None
format_car_results_stream = None
format_car_results = None

if os.path.exists(FORMATTER_PATH):
    spec = importlib.util.spec_from_file_location("formatter_loaded", FORMATTER_PATH)
    formatter = importlib.util.module_from_spec(spec)
    sys.modules["formatter_loaded"] = formatter
    spec.loader.exec_module(formatter)  # type: ignore

    # İstenilen fonksiyonları güvenli şekilde al
    format_car_results_stream = getattr(formatter, "format_car_results_stream", None)
    format_car_results = getattr(formatter, "format_car_results", None)
else:
    st.error(f"formatter.py bulunamadı: {FORMATTER_PATH}")

API_URL = "http://localhost:8000/search"

# ===========================
# Streamlit Ayarı
# ===========================
st.set_page_config(page_title="Araç Satış Chatbot", page_icon="🚗", layout="wide")
st.title("🚗 Araç Satış Asistanı")
st.markdown("Merhaba! Sana en uygun aracı bulmana yardımcı olabilirim. 😊")

# ===========================
# Sohbet Geçmişi
# ===========================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hoş geldin! Bana bütçeni, istediğin aracı ya da özellikleri sorabilirsin."}
    ]

# ===========================
# Önceki mesajları yazdır
# ===========================
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ===========================
# Kullanıcı girişi
# ===========================
if query := st.chat_input("Bir şey yaz..."):
    st.session_state["messages"].append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    # Son 3 kullanıcı mesajını history olarak gönder
    history = [m["content"] for m in st.session_state["messages"] if m["role"] == "user"][-3:]

    # FastAPI çağrısı
    try:
        with st.spinner("Aranıyor..."):
            resp = requests.post(API_URL, json={"query": query, "history": history}, timeout=30)
            resp.raise_for_status()
            cars = resp.json()
    except Exception as e:
        answer = f"⚠️ Hata: {e}"
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
        st.stop()

    # --- Yanıt üretimi ---
    if not cars:
        answer = "Sana uygun araç bulamadım. 😕 Başka bir şey sorabilirsin."
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
    else:
        # 1) Streaming fonksiyonu varsa onu kullan
        if callable(format_car_results_stream):
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in format_car_results_stream(query, cars):
                        full_response += chunk or ""
                        placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"⚠️ LLM formatlama hatası: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": full_response})

        # 2) Aksi halde tek seferde formatlayan fonksiyonu dene
        elif callable(format_car_results):
            answer = format_car_results(query, cars)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").markdown(answer)

        # 3) O da yoksa ham listeyi göster (fallback)
        else:
            lines = []
            for car in cars:
                yil = car.get("yil") or "—"
                marka = car.get("marka") or "—"
                model = car.get("model") or ""
                seri = car.get("seri") or ""
                fiyat = car.get("fiyat")
                km = car.get("kilometre")
                fiyat_str = f"{fiyat:,}".replace(",", ".") if isinstance(fiyat, (int, float)) else "bilinmiyor"
                km_str = f"{int(km):,}".replace(",", ".") if isinstance(km, (int, float)) else "bilinmiyor"
                url = car.get("url")
                desc = (
                    f"**{yil} model {marka} {model} {seri}**\n"
                    f"- Fiyat: {fiyat_str} TL\n"
                    f"- Kilometre: {km_str} km\n"
                    f"- 👉 [İlana Git]({url})" if url else ""
                )
                lines.append(desc)
            answer = "\n\n".join(lines) if lines else "Liste boş görünüyor."
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").markdown(answer)
