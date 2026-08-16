from typing import List
from PIL import Image
from transformers import AutoProcessor, AutoModel
from langchain_core.embeddings import Embeddings
import torch

_MODEL_NAME = "google/siglip-base-patch16-224"
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _as_tensor(output):
    """get_image_features()/get_text_features() return a plain tensor in
    some transformers versions and a ModelOutput wrapper (with .pooler_output
    or .image_embeds/.text_embeds) in others. Handle both instead of
    assuming one, since this differs across installed versions."""
    if torch.is_tensor(output):
        return output
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        if hasattr(output, attr):
            tensor = getattr(output, attr)
            if tensor.dim() == 3:  # last_hidden_state: (batch, seq, hidden) -> pool
                tensor = tensor.mean(dim=1)
            return tensor
    raise TypeError(f"Unexpected output type from SigLIP: {type(output)}")


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
            features = _as_tensor(model.get_image_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.squeeze(0).cpu().tolist())
        return embeddings

    @torch.no_grad()
    def embed_query(self, text: str) -> List[float]:
        """Real query text, embedded with SigLIP's text encoder.
        SigLIP's text tower caps at 64 tokens (max_position_embeddings) — it
        was trained on short captions, not long text. Your rewritten query
        can exceed that once chat history gets folded in, so truncate rather
        than let it error out."""
        model, processor = self._load()
        inputs = processor(
            text=[text],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(_device)
        features = _as_tensor(model.get_text_features(**inputs))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().tolist()