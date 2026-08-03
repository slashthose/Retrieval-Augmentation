from typing import List
from PIL import Image
from transformers import AutoProcessor, AutoModel
from langchain_core.embeddings import Embeddings
import torch

_MODEL_NAME = "google/siglip-base-patch16-224"
_device = "cuda" if torch.cuda.is_available() else "cpu"


class SiglipEmbeddings(Embeddings):
    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            self._processor = AutoProcessor.from_pretrained(_MODEL_NAME)
            self._model = AutoModel.from_pretrained(_MODEL_NAME).to(_device).eval()
        return self._model, self._processor

    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """texts here are actually image asset_paths (see module docstring)."""
        model, processor = self._load()
        embeddings = []
        for path in texts:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(_device)
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.squeeze(0).cpu().tolist())
        return embeddings

    @torch.no_grad()
    def embed_query(self, text: str) -> List[float]:
        """Real query text, embedded with SigLIP's text encoder."""
        model, processor = self._load()
        inputs = processor(text=[text], padding="max_length", return_tensors="pt").to(_device)
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()