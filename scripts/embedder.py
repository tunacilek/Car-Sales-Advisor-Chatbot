# scripts/embedder.py
from sentence_transformers import SentenceTransformer

class ST_Embedder:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    def embed_documents(self, texts): return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()
    def embed_query(self, text): return self.embed_documents([text])[0]
    def dimension(self): return self.model.get_sentence_embedding_dimension()
    def encode(self, texts):
        return self.embed_documents(texts)
