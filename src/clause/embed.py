"""Local sentence-transformers encoding through the ONNX runtime.

The baseline model is chosen for being unsurprising rather than best: it is
symmetric, so there is no asymmetric query-prefix convention to get silently
wrong. `BAAI/bge-small-en-v1.5` is the obvious upgrade at the same
dimensionality, and the entire point of building this harness first is that the
swap can then be measured instead of asserted.
"""

from collections.abc import Sequence

from huggingface_hub import scan_cache_dir
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ModelNotCachedError(Exception):
    """The model is not available locally, and this process will not download it."""


def _is_cached(model_name: str) -> bool:
    """True when the model is already in the HuggingFace cache.

    Checked before construction so an eval run cannot silently pull weights over
    the network mid-measurement: a first run that downloads has different timing
    and a different reproducibility story from every run after it.
    """
    try:
        cache = scan_cache_dir()
    except Exception:  # no cache directory yet
        return False
    wanted = model_name.lower()
    return any(repo.repo_id.lower() == wanted for repo in cache.repos)


class Encoder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        if not _is_cached(model_name):
            raise ModelNotCachedError(
                f"{model_name!r} is not in the local cache, and this process will not "
                "download it mid-measurement. Warm it first with:\n"
                f"  uv run python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{model_name}', backend='onnx')\""
            )
        # Explicit annotation: SentenceTransformer's __init__ passes through an
        # untyped compatibility decorator, so mypy infers the constructor's
        # return type as `Any` without this — silently erasing types on every
        # attribute below rather than failing anywhere near the actual cause.
        #
        # local_files_only=True: _is_cached() only proves the repo is present in
        # the HuggingFace cache, not that the specific ONNX file this load needs
        # is among the files actually on disk (a repo can have some variants
        # cached and others not). Without this flag, a cache miss on that one
        # file falls through to a silent network download mid-run — the exact
        # failure this class exists to prevent. With it, the load raises instead.
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, backend="onnx", local_files_only=True
        )
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        # get_embedding_dimension() is used instead of the brief's
        # get_sentence_embedding_dimension(): the latter is deprecated in the
        # installed sentence-transformers version, wrapped in a decorator that
        # erases its return type to `Any` for mypy, and would otherwise carry a
        # runtime FutureWarning on every call. Both return the same value.
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise ValueError(
                f"{self.model_name!r} does not declare a sentence embedding dimension"
            )
        return dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [[float(x) for x in row] for row in vectors]
