from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticEngine:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"Chargement du modèle IA '{model_name}' (cela peut prendre quelques secondes)...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list) -> list:
        """
        Transforme une liste de phrases (ex: ["Créer indu"])
        en une liste de vecteurs (embeddings).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()
