"""Optional text channel. The network was trained with 30% text dropout, so it also works without it."""
from . import config


def make_text_encoder(enabled=True):
    if not enabled:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(config.TEXT_MODEL, device="cpu")
        print(f"[text] using {config.TEXT_MODEL}")
        return lambda texts: st.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    except Exception as e:
        print(f"[text] text encoder unavailable ({type(e).__name__}); running geometry-only "
              f"(supported: the model was trained with text dropout)")
        return None
