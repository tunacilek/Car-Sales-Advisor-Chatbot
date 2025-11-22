# scripts/recommend.py
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

SYSTEM="""Aşağıdaki aday arabalar arasından kullanıcı niyetine en uygun 5 tanesini seç.
Her aracı madde halinde yaz: Başlık | Yıl | Km | Fiyat | Yakıt/Vites | Şehir | [Link]"""

prompt=ChatPromptTemplate.from_messages([("system",SYSTEM),("human","Sorgu: {query}\nAdaylar: {candidates}")])

def recommend_text(query, results, api_key, model="gpt-4o-mini"):
    cands=[{"marka":pl.get("marka"),"seri":pl.get("seri"),"model":pl.get("model"),"yil":pl.get("yil"),
            "km":pl.get("kilometre"),"fiyat":pl.get("fiyat"),"url":pl.get("url")} for _,_,pl in results[:20]]
    llm=ChatOpenAI(api_key=api_key, model=model, temperature=0)
    chain=prompt|llm|StrOutputParser()
    return chain.invoke({"query":query,"candidates":json.dumps(cands,ensure_ascii=False)})
