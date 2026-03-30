"""Knowledge card loader and vector-retrieval selector for hypothesis generation RAG.

Loads research report knowledge cards from cards.json, filters for
daily-compatible factor-construction/hypothesis cards, embeds them once,
and retrieves the most relevant cards each round based on the current
hypothesis context (trace history + RAG guidance).
"""

import json
from pathlib import Path

import numpy as np

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.oai.utils.embedding import trim_text_for_embedding

CARDS_PATH = Path(__file__).resolve().parents[4] / "cards.json"

# Keywords indicating a card requires intraday/high-frequency data
_INTRADAY_KEYWORDS = [
    "分钟", "高频", "tick", "盘口", "五档", "逐笔", "日内",
    "intraday", "order_book",
]

# Only keep cards whose layer indicates factor construction or hypothesis
_VALID_LAYERS = {"因子构建", "假设生成"}

# Module-level caches (populated once per process)
_filtered_cache: dict[str, list[dict]] | None = None
_embedding_cache: dict[str, np.ndarray] | None = None  # action -> (N, dim) matrix
_card_texts_cache: dict[str, list[str]] | None = None   # action -> list of embed texts


def _load_and_filter() -> dict[str, list[dict]]:
    """Load cards from disk, apply all filters, cache the result.

    Filtering pipeline:
      1. Remove intraday/high-frequency cards
      2. Keep only card_type in (factor, both) for factor action
      3. Must have core_idea
      4. confidence >= 0.7
      5. layer in (因子构建, 假设生成)
    """
    global _filtered_cache
    if _filtered_cache is not None:
        return _filtered_cache

    if not CARDS_PATH.exists():
        _filtered_cache = {"factor": [], "model": []}
        return _filtered_cache

    with open(CARDS_PATH, encoding="utf-8") as f:
        raw_cards = json.load(f)

    def _is_intraday(card: dict) -> bool:
        text = json.dumps(card, ensure_ascii=False).lower()
        return any(kw in text for kw in _INTRADAY_KEYWORDS)

    factor_cards = [
        c for c in raw_cards
        if not _is_intraday(c)
        and c.get("card_type") in ("factor", "both")
        and c.get("core_idea")
        and (c.get("confidence") or 0) >= 0.7
        and c.get("layer") in _VALID_LAYERS
    ]

    model_cards = [
        c for c in raw_cards
        if not _is_intraday(c)
        and c.get("card_type") in ("model", "both")
        and c.get("core_idea")
        and (c.get("confidence") or 0) >= 0.7
    ]

    _filtered_cache = {"factor": factor_cards, "model": model_cards}
    return _filtered_cache


def _card_to_embed_text(card: dict, action: str) -> str:
    """Build the text representation of a card for embedding."""
    parts = []
    cat = card.get("factor_category" if action == "factor" else "model_category", "other")
    parts.append(f"[{cat}]")
    parts.append(card.get("method_name", ""))
    parts.append(card.get("core_idea", ""))
    if card.get("steps"):
        parts.append(" ".join(card["steps"][:3]))
    return " ".join(parts)


def _get_embeddings(action: str) -> tuple[np.ndarray, list[str]]:
    """Embed all filtered cards for the given action. Cached after first call."""
    global _embedding_cache, _card_texts_cache
    if _embedding_cache is None:
        _embedding_cache = {}
        _card_texts_cache = {}

    if action in _embedding_cache:
        return _embedding_cache[action], _card_texts_cache[action]

    pool = _load_and_filter().get(action, [])
    if not pool:
        empty = np.zeros((0, 1))
        _embedding_cache[action] = empty
        _card_texts_cache[action] = []
        return empty, []

    texts = [_card_to_embed_text(c, action) for c in pool]
    logger.info(f"Embedding {len(texts)} {action} cards for vector retrieval...")

    # Batch embed via project's APIBackend (supports cache)
    BATCH_SIZE = 50
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vectors = APIBackend(use_embedding_cache=True, dump_embedding_cache=True).create_embedding(batch)
        all_vectors.extend(vectors)

    matrix = np.array(all_vectors, dtype=np.float32)
    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    _embedding_cache[action] = matrix
    _card_texts_cache[action] = texts
    logger.info(f"Embedded {len(texts)} {action} cards, dim={matrix.shape[1]}")
    return matrix, texts


def _format_card(card: dict, action: str) -> str:
    """Format a single card as a compact prompt line."""
    if action == "factor":
        cat = card.get("factor_category", "other")
        line = f"[{cat}] {card['method_name']}: {card['core_idea']}"
        if card.get("expected_ic_range") and card["expected_ic_range"] != "N/A":
            line += f" (Expected IC: {card['expected_ic_range']})"
        return "- " + line
    else:
        cat = card.get("model_category", "other")
        line = f"[{cat}] {card['method_name']}: {card['core_idea']}"
        return "- " + line


def select_cards(action: str, round_idx: int, n: int = 6, query: str = "") -> str:
    """Select n most relevant cards via vector similarity and return formatted string.

    Args:
        action: "factor" or "model"
        round_idx: current round number (len(trace.hist))
        n: number of cards to select
        query: context string to match against (e.g., RAG guidance + last feedback).
               Falls back to a generic query if empty.

    Returns:
        Formatted string of card suggestions to append to RAG, or "".
    """
    pool = _load_and_filter().get(action, [])
    if not pool:
        return ""

    # Build query from context
    if not query:
        query = f"quantitative {action} research for A-share stock market using daily OHLCV data"

    # Truncate query to fit embedding model token limit
    query = trim_text_for_embedding(query, model=LLM_SETTINGS.embedding_model)

    # Embed the query
    query_vec = np.array(
        APIBackend(use_embedding_cache=True, dump_embedding_cache=True).create_embedding(query),
        dtype=np.float32,
    )
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    # Get card embeddings
    card_matrix, _ = _get_embeddings(action)
    if card_matrix.shape[0] == 0:
        return ""

    # Cosine similarity via dot product (both are L2-normalized)
    scores = card_matrix @ query_vec
    top_indices = np.argsort(scores)[::-1][:n]

    selected = [pool[i] for i in top_indices]
    if not selected:
        return ""

    header = (
        "\n\nThe following are research-backed ideas extracted from academic and "
        "industry research reports, selected as most relevant to your current context. "
        "Use them as inspiration for your hypothesis "
        "(you only have daily OHLCV data: $open, $high, $low, $close, $volume, "
        "$vwap, so adapt ideas accordingly):"
    )
    lines = [_format_card(c, action) for c in selected]
    return header + "\n" + "\n".join(lines)
