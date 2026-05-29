"""
Second-stage re-rankers (Phase 2.1).

The hybrid retriever returns an RRF-fused pool that's only as good as the
ranks coming in. A second-stage re-rank step scores the head of that pool
against the query and reorders it — recall is set by retrieval, but
precision-at-k is set here.

Two backends ship today, both speaking the same :class:`Reranker`
protocol so the Connector can pick one at construction time:

* :class:`LLMReranker` — wraps the local Ollama chat model. Zero extra
  install, but each query costs one full LLM generation (15-30 s on a
  qwen2.5:3b-class model). Was the only option in v2.1.
* :class:`CrossEncoderReranker` — opt-in via the ``reranker`` extra.
  Loads a sentence-transformers cross-encoder (default
  ``BAAI/bge-reranker-base``) and scores in one batched forward pass.
  Sub-second per query on CPU, more accurate at the binary "is this
  passage relevant" task than asking a chat model to rate 0-10.

Both return ``list[float]`` aligned to the input passages. An empty list
is the "don't reorder, fall back to the upstream order" signal — used
on any failure path so re-rank is strictly best-effort.
"""
from __future__ import annotations

from typing import List, Optional, Protocol

from grimore.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    """Score how relevant each passage is to ``query``.

    Returns one float per input passage, higher = more relevant. An empty
    list signals "no useful scores produced" — the caller keeps the
    original order untouched.
    """
    def score(self, query: str, passages: List[str]) -> List[float]: ...


class LLMReranker:
    """Re-rank by asking the local chat model to rate 0-10.

    Single batched ``router.complete(json_format=True)`` call. Falls back
    to ``[]`` (no reorder) on any failure: circuit open, unreachable
    model, unparseable JSON, or no usable scores. The connector then
    keeps the RRF fusion order, so a flaky LLM never makes results
    worse than disabling re-rank entirely.
    """

    def __init__(self, router):
        self.router = router

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not self.router or len(passages) < 2:
            return []

        # Bound each passage at 300 chars — the model only needs a hint
        # of the content, and short prompts keep the rate at one
        # generation per query instead of pushing into a multi-minute
        # wait on slower local models.
        listing = "\n".join(
            f"[{i}] {(p or '')[:300]}" for i, p in enumerate(passages)
        )
        prompt = (
            f"Question: {query}\n\n"
            f"Passages:\n{listing}\n\n"
            "Rate how relevant each passage is to answering the question, on a "
            "0-10 scale.\n"
            'Return ONLY JSON: {"scores": [{"index": <int>, "score": <number>}, ...]}'
        )
        try:
            resp = self.router.complete(
                prompt=prompt,
                system_prompt="You rate passage relevance for retrieval re-ranking.",
                json_format=True,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("rerank_llm_failed", error=str(e))
            return []

        if not isinstance(resp, dict) or not isinstance(resp.get("scores"), list):
            return []

        # Default unscored entries to -inf so they sink below any real
        # score on a stable sort, matching the v2.1 inline behaviour.
        scores: list[float] = [float("-inf")] * len(passages)
        any_scored = False
        for entry in resp["scores"]:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("index")
            if not isinstance(idx, int) or not (0 <= idx < len(passages)):
                continue
            try:
                scores[idx] = float(entry.get("score", 0))
                any_scored = True
            except (TypeError, ValueError):
                continue
        return scores if any_scored else []


class CrossEncoderReranker:
    """Re-rank with a sentence-transformers cross-encoder.

    Construction lazy-imports ``sentence_transformers`` and raises a
    :class:`ImportError` with the install hint if the extra is missing.
    The Connector catches that and falls back to :class:`LLMReranker`,
    so a user can flip ``rerank_engine = "cross-encoder"`` in config
    even when the extra isn't installed — they just won't get the
    cross-encoder speedup until they install it.

    Default model is ``BAAI/bge-reranker-base`` (~280 MB on first load,
    cached locally afterwards). Pass ``model_name`` to swap.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            raise ImportError(
                "CrossEncoderReranker needs the 'reranker' extra. "
                "Install with: pip install 'grimore[reranker]'"
            ) from e
        self.model_name = model_name
        self._model: Optional[object] = None
        self._CrossEncoder = CrossEncoder

    def _load(self):
        # Defer the actual weight download / load to first use so
        # ``grimore`` CLI startup isn't gated on it. After the first
        # ``score`` call the warm shell pays nothing extra.
        if self._model is None:
            logger.info("cross_encoder_loading", model=self.model_name)
            self._model = self._CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages or len(passages) < 2:
            return []
        try:
            model = self._load()
            pairs = [(query, p or "") for p in passages]
            raw = model.predict(pairs)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("rerank_cross_encoder_failed", error=str(e))
            return []
        # numpy arrays / lists / tuples all support iter + float().
        try:
            return [float(s) for s in raw]
        except (TypeError, ValueError):
            return []


def build_reranker(
    engine: str,
    router,
    *,
    model_name: str = "BAAI/bge-reranker-base",
) -> Optional[Reranker]:
    """Pick a backend for the Connector.

    ``engine = "cross-encoder"`` tries that path and silently falls back
    to LLM re-rank if the extras are missing — so configuration changes
    don't break a user whose pip extras lag the config bump. ``engine =
    "llm"`` (the default) always picks :class:`LLMReranker`. Returns
    ``None`` only when neither backend is viable (no router and no
    cross-encoder extras), in which case the connector skips re-rank
    entirely.
    """
    if engine == "cross-encoder":
        try:
            return CrossEncoderReranker(model_name)
        except ImportError as e:
            logger.warning("cross_encoder_unavailable", error=str(e))
    if router is not None:
        return LLMReranker(router)
    return None
